"""Claude-powered extraction: per-video structured analysis and daily synthesis.

Design notes
- Structured outputs (``output_format=<pydantic model>``) so results are schema-valid
  without prompt gymnastics.
- Stable system prompt with a cache breakpoint; volatile content (metadata, transcript)
  goes in the user turn.
- Adaptive thinking with a configurable effort level.
- Server-side refusal fallbacks (beta) so a rare safety decline re-runs on another model
  inside the same request instead of failing the video.
- ``PROMPT_VERSION`` is stored with every analysis; bump it when the prompt or schema
  changes so the pipeline knows to re-analyse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import anthropic

from ytstock.config import Settings
from ytstock.llm_cli import ClaudeCliError, build_llm_client
from ytstock.log import get_logger
from ytstock.schemas import (
    DailySynthesis,
    TickerStats,
    UsageRecord,
    VideoAnalysis,
    VideoMeta,
)

log = get_logger(__name__)

PROMPT_VERSION = "v1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: (input, output, cache_read, cache_write)
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5": (5.0, 25.0, 0.5, 6.25),
    "claude-opus-4-8": (5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-5": (2.0, 10.0, 0.2, 2.5),
    "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25),
}

VIDEO_SYSTEM_PROMPT = """You are a buy-side research analyst who watches retail-finance YouTube \
so a portfolio team does not have to. You are given the metadata and full transcript of one \
video about the stock market. Extract what the speaker actually says, separating the speaker's \
views from your own.

Guidelines
- Attribute sentiment to the speaker; do not add your own market view.
- Tickers: uppercase, no '$'. Map indexes to ETFs (S&P 500 -> SPY, Nasdaq-100 -> QQQ, \
Dow -> DIA, Russell 2000 -> IWM). If a company is named without its ticker and you are \
confident of the ticker, use it; otherwise skip it rather than guess.
- Only record a ticker when the speaker expresses a view or gives it real attention, not \
for passing mentions in a list.
- Claims: prefer specific, checkable statements (price targets, dated predictions, cited \
data) over vague commentary. Set resolves_by only for time-bound predictions.
- Auto-generated captions contain recognition errors; infer the intended ticker/company \
from context (e.g. 'in video' -> NVDA) but flag genuinely ambiguous cases in risk_flags.
- risk_flags should name concrete concerns: sponsorship, affiliate pushes, unverifiable \
'insider' claims, extreme leverage suggestions, or a pattern of prior wrong calls the \
speaker acknowledges.
- Be concise. Summary in 3-5 sentences; rationales one sentence each."""

SYNTHESIS_SYSTEM_PROMPT = """You are a buy-side research analyst compiling a daily digest of \
what popular finance YouTubers said about the stock market. You receive one structured \
analysis per video plus ticker statistics computed across all videos. Produce a synthesis \
for a professional reader.

Guidelines
- Describe the creators' collective narrative and where they disagree. Attribute views to \
'creators' or to specific channels, never present them as your own forecast.
- Weight views by the video's view count and signal_quality, and say when a widely-viewed \
video is low quality.
- predictions_to_track must be specific and checkable (ticker/instrument, direction or \
level, horizon, source channel).
- Keep the narrative tight: two paragraphs at most."""


@dataclass
class AnalysisOutcome:
    status: str  # ok | refused | error
    result: VideoAnalysis | None
    usage: UsageRecord
    request_id: str = ""
    error: str = ""


@dataclass
class SynthesisOutcome:
    status: str
    result: DailySynthesis | None
    usage: UsageRecord
    request_id: str = ""
    error: str = ""


_TICKER_RE = re.compile(r"^[A-Z]{1,6}(?:[.-][A-Z]{1,2})?$")
_INDEX_ALIASES = {
    "SPX": "SPY",
    "S&P500": "SPY",
    "SP500": "SPY",
    "^GSPC": "SPY",
    "NDX": "QQQ",
    "^NDX": "QQQ",
    "NASDAQ": "QQQ",
    "DJI": "DIA",
    "^DJI": "DIA",
    "DOW": "DIA",
    "RUT": "IWM",
    "^RUT": "IWM",
    "VIX": "VIX",
}


def normalize_ticker(raw: str) -> str | None:
    t = raw.strip().upper().lstrip("$").replace(" ", "")
    t = _INDEX_ALIASES.get(t, t)
    return t if _TICKER_RE.match(t) else None


def normalize_analysis(analysis: VideoAnalysis) -> VideoAnalysis:
    """Post-process model output: canonical tickers, dedupe, drop junk."""
    seen: dict[str, Any] = {}
    for m in analysis.tickers:
        t = normalize_ticker(m.ticker)
        if not t:
            continue
        m.ticker = t
        prev = seen.get(t)
        if prev is None or (m.is_primary and not prev.is_primary):
            seen[t] = m
    analysis.tickers = list(seen.values())

    for lvl in analysis.price_levels:
        lvl.ticker = normalize_ticker(lvl.ticker) or lvl.ticker
    for claim in analysis.claims:
        claim.tickers = [t for t in (normalize_ticker(x) for x in claim.tickers) if t]
        if claim.resolves_by and not re.match(r"^\d{4}-\d{2}-\d{2}$", claim.resolves_by):
            claim.resolves_by = ""
    return analysis


def compute_ticker_stats(
    analyses: dict[str, VideoAnalysis],
) -> list[TickerStats]:
    """Aggregate per-ticker sentiment across videos (keyed by video_id)."""
    acc: dict[str, dict[str, Any]] = {}
    for video_id, a in analyses.items():
        for m in a.tickers:
            entry = acc.setdefault(
                m.ticker,
                {"bullish": 0, "bearish": 0, "neutral": 0, "mixed": 0, "videos": []},
            )
            entry[m.sentiment] += 1
            entry["videos"].append(video_id)
    stats = [
        TickerStats(
            ticker=t,
            mentions=len(e["videos"]),
            bullish=e["bullish"],
            bearish=e["bearish"],
            neutral=e["neutral"],
            mixed=e["mixed"],
            videos=e["videos"],
        )
        for t, e in acc.items()
    ]
    stats.sort(key=lambda s: (-s.mentions, -abs(s.net_score), s.ticker))
    return stats


def estimate_cost(model: str, usage: Any) -> float:
    price = PRICING.get(model)
    if price is None:
        return 0.0
    inp, out, cr, cw = price
    return (
        getattr(usage, "input_tokens", 0) * inp
        + getattr(usage, "output_tokens", 0) * out
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * cr
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * cw
    ) / 1_000_000


def _usage_record(model: str, response: Any) -> UsageRecord:
    u = getattr(response, "usage", None)
    served_model = getattr(response, "model", None) or model
    return UsageRecord(
        model=served_model,
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cost_usd=estimate_cost(served_model, u) if u else 0.0,
    )


def _refusal_detail(response: Any) -> str:
    details = getattr(response, "stop_details", None)
    if details is None:
        return "refusal"
    cat = getattr(details, "category", None) or "unknown"
    expl = getattr(details, "explanation", None) or ""
    return f"refusal[{cat}] {expl}".strip()


class ClaudeAnalyzer:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client if client is not None else build_llm_client(settings)

    # ------------------------------------------------------------------ #
    def _parse(self, *, system: str, user: str, output_format: type, effort: str) -> Any:
        s = self._settings
        request: dict[str, Any] = {
            "model": s.claude_model,
            "max_tokens": s.claude_max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_format": output_format,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        if s.claude_fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
            return self._client.beta.messages.parse(**request)
        return self._client.messages.parse(**request)

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_video_prompt(video: VideoMeta, transcript: str) -> str:
        published = video.published_at.isoformat(timespec="minutes")
        desc = video.description.strip()
        if len(desc) > 1500:
            desc = desc[:1500] + " …"
        return (
            "<video_metadata>\n"
            f"title: {video.title}\n"
            f"channel: {video.channel_title}\n"
            f"published_at: {published}\n"
            f"duration_seconds: {video.duration_seconds}\n"
            f"view_count: {video.view_count}\n"
            f"url: {video.url}\n"
            "</video_metadata>\n\n"
            "<description>\n"
            f"{desc}\n"
            "</description>\n\n"
            "<transcript>\n"
            f"{transcript}\n"
            "</transcript>\n\n"
            "Analyse this video according to the schema."
        )

    def analyze_video(self, video: VideoMeta, transcript: str) -> AnalysisOutcome:
        s = self._settings
        if len(transcript) > s.max_transcript_chars:
            return AnalysisOutcome(
                status="error",
                result=None,
                usage=UsageRecord(model=s.claude_model),
                error=f"transcript too long ({len(transcript)} chars > {s.max_transcript_chars})",
            )
        try:
            response = self._parse(
                system=VIDEO_SYSTEM_PROMPT,
                user=self.build_video_prompt(video, transcript),
                output_format=VideoAnalysis,
                effort=s.claude_video_effort,
            )
        except anthropic.RateLimitError as exc:
            return _api_error(s.claude_model, exc, "rate_limited")
        except anthropic.APIStatusError as exc:
            return _api_error(s.claude_model, exc, f"api_{exc.status_code}")
        except anthropic.APIConnectionError as exc:
            return _api_error(s.claude_model, exc, "connection")
        except ClaudeCliError as exc:
            return _api_error(s.claude_model, exc, "claude_cli")

        usage = _usage_record(s.claude_model, response)
        request_id = getattr(response, "_request_id", "") or ""
        if response.stop_reason == "refusal":
            return AnalysisOutcome("refused", None, usage, request_id, _refusal_detail(response))
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            return AnalysisOutcome(
                "error", None, usage, request_id, f"unparsed output (stop={response.stop_reason})"
            )
        result = normalize_analysis(response.parsed_output)
        log.info(
            "analysis.video.ok",
            video_id=video.video_id,
            tickers=len(result.tickers),
            cost_usd=round(usage.cost_usd, 4),
            served_by=usage.model,
        )
        return AnalysisOutcome("ok", result, usage, request_id)

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_synthesis_prompt(
        target_date: str,
        videos: list[VideoMeta],
        analyses: dict[str, VideoAnalysis],
        stats: list[TickerStats],
    ) -> str:
        blocks = [f"<date>{target_date}</date>", "<ticker_stats>"]
        for st in stats[:25]:
            blocks.append(
                f"{st.ticker}: mentions={st.mentions} bullish={st.bullish} "
                f"bearish={st.bearish} neutral={st.neutral} mixed={st.mixed}"
            )
        blocks.append("</ticker_stats>")
        for v in videos:
            a = analyses.get(v.video_id)
            if a is None:
                continue
            blocks.append(
                f'<video id="{v.video_id}" channel="{v.channel_title}" views="{v.view_count}" '
                f'signal_quality="{a.signal_quality}" outlook="{a.market_outlook}">\n'
                f"title: {v.title}\n"
                f"{a.model_dump_json(indent=None)}\n"
                "</video>"
            )
        blocks.append("Write the daily synthesis according to the schema.")
        return "\n".join(blocks)

    def synthesize_day(
        self,
        target_date: str,
        videos: list[VideoMeta],
        analyses: dict[str, VideoAnalysis],
        stats: list[TickerStats],
    ) -> SynthesisOutcome:
        s = self._settings
        try:
            response = self._parse(
                system=SYNTHESIS_SYSTEM_PROMPT,
                user=self.build_synthesis_prompt(target_date, videos, analyses, stats),
                output_format=DailySynthesis,
                effort=s.claude_synthesis_effort,
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError, ClaudeCliError) as exc:
            return SynthesisOutcome("error", None, UsageRecord(model=s.claude_model), "", str(exc))
        usage = _usage_record(s.claude_model, response)
        request_id = getattr(response, "_request_id", "") or ""
        if response.stop_reason == "refusal":
            return SynthesisOutcome("refused", None, usage, request_id, _refusal_detail(response))
        if response.parsed_output is None:
            return SynthesisOutcome("error", None, usage, request_id, "unparsed output")
        return SynthesisOutcome("ok", response.parsed_output, usage, request_id)


def _api_error(model: str, exc: Exception, label: str) -> AnalysisOutcome:
    request_id = getattr(exc, "request_id", "") or ""
    log.warning("analysis.video.api_error", kind=label, error=str(exc), request_id=request_id)
    return AnalysisOutcome("error", None, UsageRecord(model=model), request_id, f"{label}: {exc}")


# =========================================================================== #
# Fact-checking and trading brief (web-search grounded)
# =========================================================================== #
from ytstock.schemas import FactCheckReport, TradingBrief  # noqa: E402

FACTCHECK_PROMPT_VERSION = "v1"
BRIEF_PROMPT_VERSION = "v1"
MAX_PAUSE_TURNS = 6

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 12,
}

FACTCHECK_SYSTEM_PROMPT = """You are a fact-checker for a portfolio team. You receive a \
structured analysis of a stock-market YouTube video (claims, tickers, rationale) and the \
video's publication date. Verify the checkable claims against primary or reputable sources \
using web search.

Rules
- Check the claims that matter most to the thesis first; batch related claims into one search \
where possible. Prefer primary sources (company filings, exchange data, BLS/BEA/Fed releases, \
major financial press).
- 'supported' requires a source that states the fact; 'partially_supported' when the gist is \
right but numbers or framing are off; 'contradicted' when the sources disagree, and you must \
give the correct figure; 'unverifiable' when no source can settle it; 'not_yet_resolved' for \
predictions whose date has not passed.
- Judge claims as of the video's publication date, but add anything that changed since in \
new_context (e.g. a data release or price move after recording).
- Never invent sources. List only URLs you actually retrieved.
- Be specific and quantitative in evidence. Keep every field concise."""

BRIEF_SYSTEM_PROMPT = """You are a senior cross-asset strategist writing a trading brief for a \
discretionary trader with a horizon of the next few days to a few weeks. Your inputs are: \
structured analyses of several stock-market YouTube videos the trader selected, fact-check \
results for each, and aggregate ticker statistics. You may use web search to confirm current \
prices, the economic calendar, earnings dates, or anything else that makes the brief \
actionable; do not spend searches re-checking claims already fact-checked.

How to think
- Separate three layers: what the creators say, what the evidence shows, and what that \
implies for positioning. Weight creators by fact-check reliability, not by view count.
- Look for the non-obvious: where creators agree but the data disagrees, where a narrative is \
crowded (and therefore fragile), where a second-order effect is being ignored, where a \
catalyst is mis-dated or mispriced, and what the creators' own behaviour reveals about \
retail positioning.
- Trade ideas must be concrete: instrument, direction, catalyst, entry conditions, \
invalidation, horizon. If the evidence does not support a trade, say so; 'no trade' is a \
valid conclusion. Never fabricate prices; if you cite a level, it must come from an input \
or a search result.
- Watch-list items should be time-bound and specific (data release, earnings, a level to \
break, a scheduled speaker, an options expiry).
- Write for a professional: dense, quantitative, no filler, no generic disclaimers in the \
body (a caveats list is provided for that)."""


@dataclass
class ToolOutcome:
    status: str  # ok | refused | error
    result: Any
    usage: UsageRecord
    request_id: str = ""
    error: str = ""
    web_searches: int = 0


def _server_tool_searches(response: Any) -> int:
    u = getattr(response, "usage", None)
    st = getattr(u, "server_tool_use", None)
    return int(getattr(st, "web_search_requests", 0) or 0)


def _merge_usage(total: UsageRecord, model: str, response: Any) -> UsageRecord:
    part = _usage_record(model, response)
    return UsageRecord(
        model=part.model,
        input_tokens=total.input_tokens + part.input_tokens,
        output_tokens=total.output_tokens + part.output_tokens,
        cache_read_input_tokens=total.cache_read_input_tokens + part.cache_read_input_tokens,
        cache_creation_input_tokens=(
            total.cache_creation_input_tokens + part.cache_creation_input_tokens
        ),
        cost_usd=total.cost_usd + part.cost_usd,
    )


class GroundedAnalyzer(ClaudeAnalyzer):
    """Adds web-search-grounded calls (fact check, trading brief) to ClaudeAnalyzer."""

    def _parse_with_tools(
        self, *, system: str, user: str, output_format: type, effort: str, max_uses: int
    ) -> tuple[Any, UsageRecord, int]:
        """Like ``_parse`` but with the web search server tool, resuming on ``pause_turn``."""
        s = self._settings
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        usage = UsageRecord(model=s.claude_model)
        searches = 0
        for _ in range(MAX_PAUSE_TURNS):
            request: dict[str, Any] = {
                "model": s.claude_model,
                "max_tokens": s.claude_max_tokens,
                "system": [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                ],
                "messages": messages,
                "tools": [{**WEB_SEARCH_TOOL, "max_uses": max_uses}],
                "output_format": output_format,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }
            if s.claude_fallbacks:
                request["betas"] = [FALLBACK_BETA]
                request["fallbacks"] = "default"
                response = self._client.beta.messages.parse(**request)
            else:
                response = self._client.messages.parse(**request)
            usage = _merge_usage(usage, s.claude_model, response)
            searches += _server_tool_searches(response)
            if response.stop_reason != "pause_turn":
                return response, usage, searches
            messages.append({"role": "assistant", "content": response.content})
        return response, usage, searches  # last response; caller checks stop_reason

    def _grounded(
        self, *, system: str, user: str, output_format: type, effort: str, max_uses: int
    ) -> ToolOutcome:
        s = self._settings
        try:
            response, usage, searches = self._parse_with_tools(
                system=system,
                user=user,
                output_format=output_format,
                effort=effort,
                max_uses=max_uses,
            )
        except anthropic.APIStatusError as exc:
            return ToolOutcome(
                "error",
                None,
                UsageRecord(model=s.claude_model),
                getattr(exc, "request_id", "") or "",
                f"api_{exc.status_code}: {exc}",
            )
        except anthropic.APIConnectionError as exc:
            return ToolOutcome(
                "error", None, UsageRecord(model=s.claude_model), "", f"connection: {exc}"
            )
        except ClaudeCliError as exc:
            return ToolOutcome(
                "error", None, UsageRecord(model=s.claude_model), "", f"claude_cli: {exc}"
            )
        request_id = getattr(response, "_request_id", "") or ""
        if response.stop_reason == "refusal":
            return ToolOutcome(
                "refused", None, usage, request_id, _refusal_detail(response), searches
            )
        if response.stop_reason == "pause_turn":
            return ToolOutcome(
                "error", None, usage, request_id, "exceeded pause_turn budget", searches
            )
        if response.parsed_output is None:
            return ToolOutcome(
                "error",
                None,
                usage,
                request_id,
                f"unparsed output (stop={response.stop_reason})",
                searches,
            )
        return ToolOutcome("ok", response.parsed_output, usage, request_id, "", searches)

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_factcheck_prompt(video: VideoMeta, analysis: VideoAnalysis, today: str) -> str:
        return (
            f"<today>{today}</today>\n"
            "<video>\n"
            f"title: {video.title}\nchannel: {video.channel_title}\n"
            f"published_at: {video.published_at.date().isoformat()}\nurl: {video.url}\n"
            "</video>\n\n"
            "<analysis>\n"
            f"{analysis.model_dump_json(indent=None)}\n"
            "</analysis>\n\n"
            "Fact-check the claims and the factual basis of the market_outlook_rationale and "
            "ticker rationales. Return the report according to the schema."
        )

    def fact_check(self, video: VideoMeta, analysis: VideoAnalysis, today: str) -> ToolOutcome:
        outcome = self._grounded(
            system=FACTCHECK_SYSTEM_PROMPT,
            user=self.build_factcheck_prompt(video, analysis, today),
            output_format=FactCheckReport,
            effort=self._settings.claude_factcheck_effort,
            max_uses=self._settings.factcheck_max_searches,
        )
        log.info(
            "factcheck.done",
            video_id=video.video_id,
            status=outcome.status,
            searches=outcome.web_searches,
            cost_usd=round(outcome.usage.cost_usd, 4),
        )
        return outcome

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_brief_prompt(
        today: str,
        videos: list[VideoMeta],
        analyses: dict[str, VideoAnalysis],
        fact_checks: dict[str, FactCheckReport],
        stats: list[TickerStats],
    ) -> str:
        blocks = [f"<today>{today}</today>", "<ticker_stats>"]
        for st in stats[:30]:
            blocks.append(
                f"{st.ticker}: mentions={st.mentions} bullish={st.bullish} "
                f"bearish={st.bearish} neutral={st.neutral} mixed={st.mixed}"
            )
        blocks.append("</ticker_stats>")
        for v in videos:
            a = analyses.get(v.video_id)
            if a is None:
                continue
            fc = fact_checks.get(v.video_id)
            blocks.append(
                f'<video id="{v.video_id}" channel="{v.channel_title}" '
                f'published="{v.published_at.date().isoformat()}" views="{v.view_count}" '
                f'signal_quality="{a.signal_quality}">\n'
                f"title: {v.title}\n"
                f"<analysis>{a.model_dump_json(indent=None)}</analysis>\n"
                + (
                    f"<fact_check>{fc.model_dump_json(indent=None)}</fact_check>\n"
                    if fc
                    else "<fact_check>not available</fact_check>\n"
                )
                + "</video>"
            )
        blocks.append("Write the trading brief according to the schema.")
        return "\n".join(blocks)

    def trading_brief(
        self,
        today: str,
        videos: list[VideoMeta],
        analyses: dict[str, VideoAnalysis],
        fact_checks: dict[str, FactCheckReport],
        stats: list[TickerStats],
    ) -> ToolOutcome:
        outcome = self._grounded(
            system=BRIEF_SYSTEM_PROMPT,
            user=self.build_brief_prompt(today, videos, analyses, fact_checks, stats),
            output_format=TradingBrief,
            effort=self._settings.claude_brief_effort,
            max_uses=self._settings.brief_max_searches,
        )
        log.info(
            "brief.done",
            status=outcome.status,
            searches=outcome.web_searches,
            cost_usd=round(outcome.usage.cost_usd, 4),
        )
        return outcome
