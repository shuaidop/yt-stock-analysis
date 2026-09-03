from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ytstock.config import Settings
from ytstock.db import Database
from ytstock.schemas import DailySynthesis, FactCheckReport, TradingBrief, VideoAnalysis, VideoMeta


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        youtube_api_key="test-key",
        anthropic_api_key="test-key",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        reports_dir=tmp_path / "reports",
        sources_file=Path(__file__).parent.parent / "configs" / "sources.yaml",
        min_view_count=100,
        analysis_concurrency=2,
    )


@pytest.fixture
def db(settings: Settings) -> Database:
    return Database(settings.database_url)


def make_video(video_id: str = "abc123def45", **kw: Any) -> VideoMeta:
    base: dict[str, Any] = {
        "video_id": video_id,
        "title": "Stock Market Today: NVDA rips, Fed on deck",
        "channel_id": "UCxxxxxxxxxxxxxxxxxxxxxx",
        "channel_title": "Test Channel",
        "published_at": datetime(2026, 9, 2, 20, 30, tzinfo=UTC),
        "duration_seconds": 900,
        "view_count": 50_000,
        "like_count": 1000,
        "comment_count": 100,
        "description": "Daily recap",
        "discovery_source": "search:stock market today",
    }
    base.update(kw)
    return VideoMeta(**base)


@pytest.fixture
def video() -> VideoMeta:
    return make_video()


ANALYSIS_DICT: dict[str, Any] = {
    "summary": "The host reviews a strong session led by semis and previews the FOMC.",
    "market_outlook": "bullish",
    "market_outlook_rationale": "Breadth improved and the host expects a dovish Fed.",
    "key_themes": ["AI capex", "FOMC"],
    "macro_events": ["FOMC decision"],
    "tickers": [
        {
            "ticker": "$nvda",
            "name": "Nvidia",
            "sentiment": "bullish",
            "confidence": "high",
            "timeframe": "weeks",
            "rationale": "Data-center demand keeps accelerating.",
            "is_primary": True,
        },
        {
            "ticker": "SPX",
            "name": "S&P 500",
            "sentiment": "bullish",
            "confidence": "medium",
            "timeframe": "days",
            "rationale": "Above the 50-day.",
            "is_primary": False,
        },
        {
            "ticker": "NVDA",
            "name": "Nvidia (again)",
            "sentiment": "neutral",
            "confidence": "low",
            "timeframe": "unspecified",
            "rationale": "Passing mention.",
            "is_primary": False,
        },
        {
            "ticker": "not a ticker",
            "name": "junk",
            "sentiment": "neutral",
            "confidence": "low",
            "timeframe": "unspecified",
            "rationale": "junk",
            "is_primary": False,
        },
    ],
    "price_levels": [{"ticker": "spx", "level": "5,400", "kind": "support"}],
    "claims": [
        {
            "text": "NVDA will hit $200 by year end.",
            "kind": "prediction",
            "tickers": ["nvda"],
            "resolves_by": "2026-12-31",
        },
        {
            "text": "CPI came in at 2.9% last month.",
            "kind": "factual_claim",
            "tickers": [],
            "resolves_by": "soon",
        },
    ],
    "recommended_actions": ["Buy the dip in semis."],
    "risk_flags": ["Sponsored segment for a brokerage."],
    "signal_quality": 3,
}

SYNTHESIS_DICT: dict[str, Any] = {
    "headline": "Creators lean bullish into the Fed",
    "market_narrative": "Most creators expect a dovish outcome.",
    "consensus_outlook": "bullish",
    "agreement_level": "moderate",
    "top_tickers": [{"ticker": "NVDA", "consensus": "bullish", "note": "Everyone likes it."}],
    "contrarian_views": ["One channel warns of a blow-off top."],
    "upcoming_catalysts": ["FOMC on Wednesday"],
    "predictions_to_track": ["NVDA $200 by 2026-12-31 (Test Channel)"],
}

FACTCHECK_DICT: dict[str, Any] = {
    "verifications": [
        {
            "claim": "CPI came in at 2.9% last month.",
            "verdict": "contradicted",
            "evidence": "BLS reported 2.7% headline CPI.",
            "sources": ["https://www.bls.gov/cpi/"],
            "materiality": "medium",
        }
    ],
    "reliability_summary": "Mostly right on direction, sloppy on numbers.",
    "reliability_score": 3,
    "corrections": ["CPI was 2.7%, not 2.9%."],
    "new_context": ["Jobs report released after recording was weaker than expected."],
}

BRIEF_DICT: dict[str, Any] = {
    "headline": "Long semis into the Fed, hedge with puts",
    "executive_summary": "Creators are uniformly bullish; the data is less clear.",
    "macro_economy": "Inflation is cooling slower than creators claim.",
    "equity_market": "SPX holding the 50-day.",
    "sectors_and_stocks": "Semis lead; NVDA is the consensus long.",
    "derivatives_and_flows": "Not discussed.",
    "hidden_logic": ["Consensus bullishness among retail creators is a crowding signal."],
    "trade_ideas": [
        {
            "title": "NVDA pre-Fed drift",
            "ticker": "NVDA",
            "instrument": "stock",
            "direction": "long",
            "thesis": "Momentum plus positive revisions.",
            "source_basis": "Test Channel; fact check confirmed guidance numbers.",
            "catalyst": "FOMC",
            "entry_conditions": "Holds above 180 into Wednesday.",
            "invalidation": "Close below 175.",
            "horizon": "days",
            "conviction": "medium",
            "key_risks": ["Hawkish surprise"],
        }
    ],
    "watch_list": [
        {
            "what": "FOMC statement",
            "when": "Wednesday 14:00 ET",
            "why_it_matters": "Rate path drives multiples.",
            "tickers": ["SPY", "QQQ"],
        }
    ],
    "creator_reliability": "Test Channel: 3/5.",
    "disagreements": ["Creators vs. BLS on CPI."],
    "risks_and_caveats": ["Small sample of videos."],
}


def fake_response(
    parsed: Any, *, stop_reason: str = "end_turn", model: str = "claude-opus-5", searches: int = 0
):
    return SimpleNamespace(
        stop_reason=stop_reason,
        parsed_output=parsed,
        content=[],
        model=model,
        stop_details=None,
        _request_id="req_test",
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=0,
            server_tool_use=SimpleNamespace(web_search_requests=searches),
        ),
    )


class FakeClaudeClient:
    """Duck-types client.messages.parse / client.beta.messages.parse."""

    def __init__(self, handler=None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._handler = handler or self._default
        self.messages = SimpleNamespace(parse=self._parse)
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **request: Any):
        self.calls.append(request)
        return self._handler(request)

    @staticmethod
    def _default(request: dict[str, Any]):
        fmt = request["output_format"]
        if fmt is VideoAnalysis:
            return fake_response(VideoAnalysis.model_validate(ANALYSIS_DICT))
        if fmt is DailySynthesis:
            return fake_response(DailySynthesis.model_validate(SYNTHESIS_DICT))
        if fmt is FactCheckReport:
            return fake_response(FactCheckReport.model_validate(FACTCHECK_DICT), searches=3)
        if fmt is TradingBrief:
            return fake_response(TradingBrief.model_validate(BRIEF_DICT), searches=2)
        raise AssertionError(f"unexpected output_format {fmt}")


@pytest.fixture
def fake_client() -> FakeClaudeClient:
    return FakeClaudeClient()
