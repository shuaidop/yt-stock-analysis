"""Pydantic models that cross module boundaries.

``VideoAnalysis`` and ``DailySynthesis`` double as the JSON schemas handed to
Claude's structured-output mode, so they deliberately avoid numeric range
constraints (unsupported there) and use ``Literal`` enums instead.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["bullish", "bearish", "neutral", "mixed"]
Confidence = Literal["low", "medium", "high"]
Timeframe = Literal["intraday", "days", "weeks", "months", "long_term", "unspecified"]


# --------------------------------------------------------------------------- #
# Discovery / transcripts
# --------------------------------------------------------------------------- #
class VideoMeta(BaseModel):
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    published_at: datetime
    duration_seconds: int
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    description: str = ""
    discovery_source: str = ""  # e.g. "search:stock market today" / "channel:UC..."

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


class TranscriptResult(BaseModel):
    video_id: str
    text: str
    language: str
    source: Literal["youtube_captions", "youtube_captions_translated", "whisper"]
    is_generated: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------- #
# Per-video analysis (Claude structured output)
# --------------------------------------------------------------------------- #
class PriceLevel(BaseModel):
    ticker: str
    level: str = Field(description="Price as stated, e.g. '185' or '5,400-5,450'.")
    kind: Literal["support", "resistance", "target", "stop", "entry", "other"]


class TickerMention(BaseModel):
    ticker: str = Field(
        description="Uppercase ticker symbol without '$'. Use index ETFs for broad indexes "
        "(SPY for S&P 500, QQQ for Nasdaq-100, DIA for Dow, IWM for Russell 2000)."
    )
    name: str = Field(description="Company or instrument name as referenced in the video.")
    sentiment: Sentiment
    confidence: Confidence = Field(description="How clearly the speaker commits to the view.")
    timeframe: Timeframe
    rationale: str = Field(description="One sentence: why the speaker holds this view.")
    is_primary: bool = Field(description="True if this ticker is a main focus of the video.")


class Claim(BaseModel):
    text: str = Field(description="The claim, paraphrased concisely and faithfully.")
    kind: Literal["prediction", "factual_claim", "opinion", "recommendation"]
    tickers: list[str] = Field(description="Tickers the claim concerns; empty if macro.")
    resolves_by: str = Field(
        description="ISO date (YYYY-MM-DD) by which a prediction can be checked, or '' if "
        "not a time-bound prediction."
    )


class VideoAnalysis(BaseModel):
    summary: str = Field(description="3-5 sentence neutral summary of the video's content.")
    market_outlook: Sentiment = Field(description="The speaker's overall stance on the market.")
    market_outlook_rationale: str
    key_themes: list[str] = Field(description="3-8 short phrases, e.g. 'AI capex slowdown'.")
    macro_events: list[str] = Field(
        description="Scheduled or recent macro events discussed (FOMC, CPI, earnings, etc.)."
    )
    tickers: list[TickerMention]
    price_levels: list[PriceLevel]
    claims: list[Claim] = Field(description="Most consequential claims, at most 10.")
    recommended_actions: list[str] = Field(
        description="Concrete actions the speaker suggests viewers take, verbatim-ish."
    )
    risk_flags: list[str] = Field(
        description="Signals of hype, sponsorship, unverifiable claims, or conflicts of interest."
    )
    signal_quality: Literal[1, 2, 3, 4, 5] = Field(
        description="1 = pure hype/entertainment, 5 = well-reasoned, sourced analysis."
    )


# --------------------------------------------------------------------------- #
# Daily cross-video synthesis
# --------------------------------------------------------------------------- #
class TickerConsensus(BaseModel):
    ticker: str
    consensus: Sentiment
    note: str = Field(description="One sentence on the shared or conflicting view.")


class DailySynthesis(BaseModel):
    headline: str = Field(description="One-line headline for the day's narrative.")
    market_narrative: str = Field(description="1-2 paragraphs synthesising what creators said.")
    consensus_outlook: Sentiment
    agreement_level: Literal["strong", "moderate", "split"]
    top_tickers: list[TickerConsensus] = Field(description="Most-discussed tickers, up to 10.")
    contrarian_views: list[str] = Field(description="Views that cut against the majority.")
    upcoming_catalysts: list[str]
    predictions_to_track: list[str] = Field(
        description="Specific, checkable predictions with the ticker and stated horizon."
    )


# --------------------------------------------------------------------------- #
# Aggregates computed in code (not by the model)
# --------------------------------------------------------------------------- #
class TickerStats(BaseModel):
    ticker: str
    mentions: int
    bullish: int
    bearish: int
    neutral: int
    mixed: int
    videos: list[str]

    @property
    def net_score(self) -> int:
        return self.bullish - self.bearish


class UsageRecord(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0


class RunSummary(BaseModel):
    target_date: date
    discovered: int = 0
    transcribed: int = 0
    transcript_failures: int = 0
    analyzed: int = 0
    analysis_failures: int = 0
    report_path: str | None = None
    total_cost_usd: float = 0.0


# --------------------------------------------------------------------------- #
# Fact-checking (Claude + web search)
# --------------------------------------------------------------------------- #
Verdict = Literal[
    "supported", "partially_supported", "contradicted", "unverifiable", "not_yet_resolved"
]


class ClaimVerification(BaseModel):
    claim: str = Field(description="The claim being checked, as extracted from the video.")
    verdict: Verdict
    evidence: str = Field(
        description="1-3 sentences: what the sources actually show, with the specific numbers "
        "or facts. For contradicted claims state the correct figure."
    )
    sources: list[str] = Field(description="URLs consulted; empty if unverifiable.")
    materiality: Confidence = Field(
        description="How much this claim matters to the video's thesis (low/medium/high)."
    )


class FactCheckReport(BaseModel):
    verifications: list[ClaimVerification]
    reliability_summary: str = Field(
        description="2-3 sentences on how factually reliable this video is overall."
    )
    reliability_score: Literal[1, 2, 3, 4, 5] = Field(
        description="1 = mostly wrong or unverifiable, 5 = accurate and well-sourced."
    )
    corrections: list[str] = Field(
        description="Material errors in the video with the correct information."
    )
    new_context: list[str] = Field(
        description="Relevant facts found while checking that the video did not mention "
        "(e.g. data released after recording)."
    )


# --------------------------------------------------------------------------- #
# Trading brief (final synthesis across videos + fact checks + live context)
# --------------------------------------------------------------------------- #
Direction = Literal["long", "short", "neutral", "hedge"]
Instrument = Literal[
    "stock", "etf", "call_option", "put_option", "option_spread", "futures", "bond", "other"
]


class TradeIdea(BaseModel):
    title: str = Field(description="Short label, e.g. 'NVDA post-earnings drift'.")
    ticker: str
    instrument: Instrument
    direction: Direction
    thesis: str = Field(description="2-4 sentences: the reasoning chain behind the idea.")
    source_basis: str = Field(
        description="Which video(s)/claims this comes from and what fact-checking showed."
    )
    catalyst: str = Field(description="The event or condition expected to move the trade.")
    entry_conditions: str = Field(description="What must be true before acting; be concrete.")
    invalidation: str = Field(description="What would prove the idea wrong (level/event/date).")
    horizon: Timeframe
    conviction: Confidence
    key_risks: list[str]


class WatchItem(BaseModel):
    what: str = Field(description="The data release, event, level, or behaviour to watch.")
    when: str = Field(description="Date/time, or relative timing like 'Thursday pre-market'.")
    why_it_matters: str
    tickers: list[str]


class TradingBrief(BaseModel):
    headline: str
    executive_summary: str = Field(description="One paragraph a PM can read in 30 seconds.")
    macro_economy: str = Field(
        description="Growth, inflation, rates, Fed, fiscal, geopolitics: what the videos "
        "claim, what checks out, and what it implies."
    )
    equity_market: str = Field(
        description="Index levels, breadth, positioning, technical levels, seasonality."
    )
    sectors_and_stocks: str = Field(description="Single-name and sector theses discussed.")
    derivatives_and_flows: str = Field(
        description="Options positioning, implied vol, gamma, futures, rates, FX, commodities "
        "where relevant. Say 'not discussed' if nothing was said."
    )
    hidden_logic: list[str] = Field(
        description="Non-obvious second-order ideas: what the creators' collective behaviour "
        "or blind spots imply, crowded trades, mispriced narratives, reflexive setups."
    )
    trade_ideas: list[TradeIdea] = Field(description="Ranked by conviction; at most 8.")
    watch_list: list[WatchItem] = Field(description="Things to monitor over the next few days.")
    creator_reliability: str = Field(
        description="How much to trust each source based on the fact checks."
    )
    disagreements: list[str] = Field(description="Where creators or the evidence conflict.")
    risks_and_caveats: list[str]
