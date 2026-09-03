from __future__ import annotations

from datetime import date

from ytstock.analysis import compute_ticker_stats, normalize_analysis
from ytstock.report import (
    build_brief_json,
    build_json,
    render_brief_markdown,
    render_markdown,
    write_brief,
    write_report,
)
from ytstock.schemas import DailySynthesis, FactCheckReport, TradingBrief, VideoAnalysis

from .conftest import ANALYSIS_DICT, BRIEF_DICT, FACTCHECK_DICT, SYNTHESIS_DICT, make_video


def test_render_daily_markdown_and_json(tmp_path):
    v1, v2 = make_video("v0000000001"), make_video("v0000000002", title="Failed one")
    a = normalize_analysis(VideoAnalysis.model_validate(ANALYSIS_DICT))
    analyses = {v1.video_id: a}
    stats = compute_ticker_stats(analyses)
    syn = DailySynthesis.model_validate(SYNTHESIS_DICT)
    md = render_markdown(
        date(2026, 9, 2),
        [v1, v2],
        analyses,
        stats,
        syn,
        failures={v2.video_id: "transcript unavailable"},
        cost_usd=1.234,
    )
    assert "# Stock-market YouTube digest — 2026-09-02" in md
    assert "Creators lean bullish into the Fed" in md
    assert "| NVDA | 1 |" in md
    assert "_Skipped: transcript unavailable_" in md
    assert "$1.23" in md
    payload = build_json(date(2026, 9, 2), [v1, v2], analyses, stats, syn, cost_usd=1.234)
    assert payload["videos"][1]["analysis"] is None and payload["synthesis"]["headline"]
    path = write_report(tmp_path / "r", date(2026, 9, 2), md, payload)
    assert path.exists() and (tmp_path / "r" / "latest.md").read_text() == md
    assert (tmp_path / "r" / "2026-09-02.json").exists()


def test_render_brief_markdown(tmp_path):
    v = make_video("v0000000001")
    a = normalize_analysis(VideoAnalysis.model_validate(ANALYSIS_DICT))
    fc = FactCheckReport.model_validate(FACTCHECK_DICT)
    brief = TradingBrief.model_validate(BRIEF_DICT)
    stats = compute_ticker_stats({v.video_id: a})
    md = render_brief_markdown(
        date(2026, 9, 2), [v], {v.video_id: a}, {v.video_id: fc}, stats, brief, cost_usd=2.0
    )
    for needle in (
        "# Trading brief — 2026-09-02",
        "### Hidden logic / second-order ideas",
        "#### 1. NVDA pre-Fed drift (NVDA, long via stock)",
        "**Invalidation:** Close below 175.",
        "### Watch list (next few days)",
        "❌ contradicted",
        "[1](https://www.bls.gov/cpi/)",
        "**Corrections:**",
        "reliability 3/5",
    ):
        assert needle in md, needle
    payload = build_brief_json(
        date(2026, 9, 2), [v], {v.video_id: a}, {v.video_id: fc}, stats, brief
    )
    assert payload["brief"]["trade_ideas"][0]["ticker"] == "NVDA"
    assert payload["videos"][0]["fact_check"]["reliability_score"] == 3
    path = write_brief(tmp_path / "r", date(2026, 9, 2), md, payload)
    assert path.name == "brief-2026-09-02.md" and (tmp_path / "r" / "latest-brief.md").exists()


def test_brief_without_result_still_renders():
    v = make_video()
    md = render_brief_markdown(
        date(2026, 9, 2), [v], {}, {}, [], None, failures={v.video_id: "no transcript yet"}
    )
    assert "_Skipped: no transcript yet_" in md
