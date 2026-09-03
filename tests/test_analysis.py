from __future__ import annotations

from types import SimpleNamespace

from ytstock.analysis import (
    FALLBACK_BETA,
    ClaudeAnalyzer,
    GroundedAnalyzer,
    compute_ticker_stats,
    estimate_cost,
    normalize_analysis,
    normalize_ticker,
)
from ytstock.schemas import FactCheckReport, TradingBrief, VideoAnalysis

from .conftest import ANALYSIS_DICT, FakeClaudeClient, fake_response


def test_normalize_ticker():
    assert normalize_ticker("$nvda") == "NVDA"
    assert normalize_ticker("SPX") == "SPY"
    assert normalize_ticker("brk.b") == "BRK.B"
    assert normalize_ticker("not a ticker") is None
    assert normalize_ticker("") is None


def test_normalize_analysis_dedupes_and_cleans():
    a = normalize_analysis(VideoAnalysis.model_validate(ANALYSIS_DICT))
    tickers = {m.ticker: m for m in a.tickers}
    assert set(tickers) == {"NVDA", "SPY"}
    assert tickers["NVDA"].is_primary  # primary kept over the duplicate passing mention
    assert a.price_levels[0].ticker == "SPY"
    assert a.claims[0].tickers == ["NVDA"]
    assert a.claims[1].resolves_by == ""  # 'soon' is not a date


def test_ticker_stats_ordering():
    a = normalize_analysis(VideoAnalysis.model_validate(ANALYSIS_DICT))
    b = VideoAnalysis.model_validate(
        {
            **ANALYSIS_DICT,
            "tickers": [
                {**ANALYSIS_DICT["tickers"][0], "sentiment": "bearish"},
            ],
        }
    )
    stats = compute_ticker_stats({"v1": a, "v2": normalize_analysis(b)})
    assert stats[0].ticker == "NVDA" and stats[0].mentions == 2
    assert stats[0].bullish == 1 and stats[0].bearish == 1 and stats[0].net_score == 0
    assert stats[1].ticker == "SPY"


def test_estimate_cost_uses_pricing():
    usage = SimpleNamespace(
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    assert estimate_cost("claude-opus-5", usage) == 5.0
    assert estimate_cost("unknown-model", usage) == 0.0


def test_analyze_video_happy_path(settings, video, fake_client):
    analyzer = ClaudeAnalyzer(settings, client=fake_client)
    out = analyzer.analyze_video(video, "transcript text")
    assert out.status == "ok" and out.result is not None
    assert {m.ticker for m in out.result.tickers} == {"NVDA", "SPY"}
    assert out.usage.cost_usd > 0 and out.request_id == "req_test"

    req = fake_client.calls[0]
    assert req["model"] == "claude-opus-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": settings.claude_video_effort}
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "<transcript>" in req["messages"][0]["content"]


def test_analyze_video_without_fallbacks_uses_non_beta(settings, video):
    settings = settings.model_copy(update={"claude_fallbacks": False})
    calls = []
    client = FakeClaudeClient()
    client.beta.messages.parse = lambda **r: (_ for _ in ()).throw(AssertionError("beta used"))
    client.messages.parse = lambda **r: (
        calls.append(r) or fake_response(VideoAnalysis.model_validate(ANALYSIS_DICT))
    )
    out = ClaudeAnalyzer(settings, client=client).analyze_video(video, "t")
    assert out.status == "ok" and "betas" not in calls[0]


def test_analyze_video_refusal_and_truncation(settings, video):
    refused = FakeClaudeClient(lambda r: fake_response(None, stop_reason="refusal"))
    out = ClaudeAnalyzer(settings, client=refused).analyze_video(video, "t")
    assert out.status == "refused" and "refusal" in out.error

    cut = FakeClaudeClient(lambda r: fake_response(None, stop_reason="max_tokens"))
    out = ClaudeAnalyzer(settings, client=cut).analyze_video(video, "t")
    assert out.status == "error" and "max_tokens" in out.error


def test_analyze_video_rejects_oversized_transcript(settings, video, fake_client):
    settings = settings.model_copy(update={"max_transcript_chars": 10})
    out = ClaudeAnalyzer(settings, client=fake_client).analyze_video(video, "x" * 11)
    assert out.status == "error" and "too long" in out.error and not fake_client.calls


def test_api_errors_become_outcomes(settings, video):
    import anthropic
    import httpx2

    def boom(_):
        req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.RateLimitError(
            "slow down", response=httpx2.Response(429, request=req), body=None
        )

    out = ClaudeAnalyzer(settings, client=FakeClaudeClient(boom)).analyze_video(video, "t")
    assert out.status == "error" and out.error.startswith("rate_limited")


def test_grounded_factcheck_passes_web_search_and_resumes_pause_turn(settings, video):
    analysis = VideoAnalysis.model_validate(ANALYSIS_DICT)
    n = {"calls": 0}

    def handler(request):
        n["calls"] += 1
        assert request["tools"][0]["type"].startswith("web_search")
        if n["calls"] == 1:
            r = fake_response(None, stop_reason="pause_turn", searches=2)
            r.content = [{"type": "text", "text": "partial"}]
            return r
        assert request["messages"][1]["role"] == "assistant"
        from .conftest import FACTCHECK_DICT

        return fake_response(FactCheckReport.model_validate(FACTCHECK_DICT), searches=1)

    out = GroundedAnalyzer(settings, client=FakeClaudeClient(handler)).fact_check(
        video, analysis, "2026-09-02"
    )
    assert out.status == "ok" and n["calls"] == 2
    assert out.web_searches == 3
    assert out.usage.input_tokens == 2000  # merged across both turns


def test_trading_brief_prompt_includes_fact_checks(settings, video, fake_client):
    from .conftest import FACTCHECK_DICT

    analysis = VideoAnalysis.model_validate(ANALYSIS_DICT)
    fc = FactCheckReport.model_validate(FACTCHECK_DICT)
    stats = compute_ticker_stats({video.video_id: analysis})
    out = GroundedAnalyzer(settings, client=fake_client).trading_brief(
        "2026-09-02", [video], {video.video_id: analysis}, {video.video_id: fc}, stats
    )
    assert out.status == "ok" and isinstance(out.result, TradingBrief)
    prompt = fake_client.calls[0]["messages"][0]["content"]
    assert "<fact_check>" in prompt and "reliability_score" in prompt
    assert fake_client.calls[0]["output_config"] == {"effort": settings.claude_brief_effort}
