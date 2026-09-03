"""Render the daily report as Markdown and JSON."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ytstock.schemas import DailySynthesis, TickerStats, VideoAnalysis, VideoMeta

_ARROW = {
    "bullish": "▲ bullish",
    "bearish": "▼ bearish",
    "neutral": "— neutral",
    "mixed": "◆ mixed",
}


def _fmt_views(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def render_markdown(
    target_date: date,
    videos: list[VideoMeta],
    analyses: dict[str, VideoAnalysis],
    stats: list[TickerStats],
    synthesis: DailySynthesis | None,
    *,
    failures: dict[str, str] | None = None,
    cost_usd: float = 0.0,
) -> str:
    lines: list[str] = [f"# Stock-market YouTube digest — {target_date.isoformat()}", ""]
    analysed = [v for v in videos if v.video_id in analyses]
    lines.append(
        f"{len(analysed)} of {len(videos)} videos analysed · "
        f"{sum(v.view_count for v in analysed):,} combined views · "
        f"est. cost ${cost_usd:.2f}"
    )
    lines.append("")

    if synthesis:
        lines += [
            f"## {synthesis.headline}",
            "",
            f"**Consensus:** {_ARROW.get(synthesis.consensus_outlook, synthesis.consensus_outlook)}"
            f" ({synthesis.agreement_level} agreement)",
            "",
            synthesis.market_narrative,
            "",
        ]
        if synthesis.top_tickers:
            lines += ["### Ticker consensus", "", "| Ticker | View | Note |", "|---|---|---|"]
            for t in synthesis.top_tickers:
                lines.append(
                    f"| **{t.ticker}** | {_ARROW.get(t.consensus, t.consensus)} | {t.note} |"
                )
            lines.append("")
        if synthesis.contrarian_views:
            lines += ["### Contrarian views", ""] + [f"- {c}" for c in synthesis.contrarian_views]
            lines.append("")
        if synthesis.upcoming_catalysts:
            lines += ["### Upcoming catalysts", ""] + [
                f"- {c}" for c in synthesis.upcoming_catalysts
            ]
            lines.append("")
        if synthesis.predictions_to_track:
            lines += ["### Predictions to track", ""] + [
                f"- {p}" for p in synthesis.predictions_to_track
            ]
            lines.append("")

    if stats:
        lines += [
            "## Ticker mentions",
            "",
            "| Ticker | Mentions | Bullish | Bearish | Neutral/Mixed | Net |",
            "|---|---|---|---|---|---|",
        ]
        for st in stats[:20]:
            lines.append(
                f"| {st.ticker} | {st.mentions} | {st.bullish} | {st.bearish} | "
                f"{st.neutral + st.mixed} | {st.net_score:+d} |"
            )
        lines.append("")

    lines += ["## Videos", ""]
    for i, v in enumerate(videos, 1):
        a = analyses.get(v.video_id)
        lines.append(f"### {i}. [{v.title}]({v.url})")
        lines.append(
            f"*{v.channel_title}* · {_fmt_views(v.view_count)} views · "
            f"{_duration(v.duration_seconds)}"
            + (
                f" · {_ARROW.get(a.market_outlook, a.market_outlook)} · signal {a.signal_quality}/5"
                if a
                else ""
            )
        )
        lines.append("")
        if a is None:
            reason = (failures or {}).get(v.video_id, "not analysed")
            lines += [f"_Skipped: {reason}_", ""]
            continue
        lines += [a.summary, ""]
        if a.key_themes:
            lines.append("**Themes:** " + ", ".join(a.key_themes))
        if a.tickers:
            primary = [m for m in a.tickers if m.is_primary] or a.tickers
            lines.append(
                "**Tickers:** "
                + ", ".join(f"{m.ticker} ({m.sentiment}, {m.timeframe})" for m in primary[:8])
            )
        if a.claims:
            lines += ["", "**Key claims:**"]
            for c in a.claims[:5]:
                suffix = f" _(by {c.resolves_by})_" if c.resolves_by else ""
                tick = f" [{', '.join(c.tickers)}]" if c.tickers else ""
                lines.append(f"- ({c.kind}) {c.text}{tick}{suffix}")
        if a.risk_flags:
            lines += ["", "**Risk flags:** " + "; ".join(a.risk_flags)]
        lines.append("")

    lines.append("---")
    lines.append(
        "_Generated automatically from public YouTube transcripts. Views are attributed to the "
        "creators and are not investment advice._"
    )
    return "\n".join(lines) + "\n"


def build_json(
    target_date: date,
    videos: list[VideoMeta],
    analyses: dict[str, VideoAnalysis],
    stats: list[TickerStats],
    synthesis: DailySynthesis | None,
    *,
    failures: dict[str, str] | None = None,
    cost_usd: float = 0.0,
) -> dict:
    return {
        "date": target_date.isoformat(),
        "generated_cost_usd": round(cost_usd, 4),
        "synthesis": synthesis.model_dump() if synthesis else None,
        "ticker_stats": [s.model_dump() for s in stats],
        "videos": [
            {
                **v.model_dump(mode="json"),
                "url": v.url,
                "analysis": analyses[v.video_id].model_dump() if v.video_id in analyses else None,
                "failure": (failures or {}).get(v.video_id),
            }
            for v in videos
        ],
    }


def write_report(reports_dir: Path, target_date: date, markdown: str, payload: dict) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{target_date.isoformat()}.md"
    md_path.write_text(markdown, encoding="utf-8")
    (reports_dir / f"{target_date.isoformat()}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (reports_dir / "latest.md").write_text(markdown, encoding="utf-8")
    return md_path


# --------------------------------------------------------------------------- #
# Brief mode (explicit URLs + fact checks + trading brief)
# --------------------------------------------------------------------------- #
from ytstock.schemas import FactCheckReport, TradingBrief  # noqa: E402

_VERDICT = {
    "supported": "✅ supported",
    "partially_supported": "🟡 partial",
    "contradicted": "❌ contradicted",
    "unverifiable": "❔ unverifiable",
    "not_yet_resolved": "⏳ pending",
}


def render_brief_markdown(
    target_date: date,
    videos: list[VideoMeta],
    analyses: dict[str, VideoAnalysis],
    fact_checks: dict[str, FactCheckReport],
    stats: list[TickerStats],
    brief: TradingBrief | None,
    *,
    failures: dict[str, str] | None = None,
    cost_usd: float = 0.0,
) -> str:
    L: list[str] = [f"# Trading brief — {target_date.isoformat()}", ""]
    L.append(
        f"{len(analyses)} of {len(videos)} videos analysed · "
        f"{len(fact_checks)} fact-checked · est. cost ${cost_usd:.2f}"
    )
    L.append("")

    if brief:
        L += [f"## {brief.headline}", "", brief.executive_summary, ""]
        L += ["### Macro / economy", "", brief.macro_economy, ""]
        L += ["### Equity market", "", brief.equity_market, ""]
        L += ["### Sectors and single names", "", brief.sectors_and_stocks, ""]
        L += ["### Derivatives, rates and flows", "", brief.derivatives_and_flows, ""]
        if brief.hidden_logic:
            L += ["### Hidden logic / second-order ideas", ""]
            L += [f"- {h}" for h in brief.hidden_logic] + [""]
        if brief.trade_ideas:
            L += [
                "### Trade ideas",
                "",
                "| # | Idea | Ticker | Instrument | Dir | Horizon | Conviction |",
                "|---|---|---|---|---|---|---|",
            ]
            for i, t in enumerate(brief.trade_ideas, 1):
                L.append(
                    f"| {i} | {t.title} | {t.ticker} | {t.instrument} | {t.direction} | "
                    f"{t.horizon} | {t.conviction} |"
                )
            L.append("")
            for i, t in enumerate(brief.trade_ideas, 1):
                L += [
                    f"#### {i}. {t.title} ({t.ticker}, {t.direction} via {t.instrument})",
                    "",
                    t.thesis,
                    "",
                    f"- **Basis:** {t.source_basis}",
                    f"- **Catalyst:** {t.catalyst}",
                    f"- **Entry conditions:** {t.entry_conditions}",
                    f"- **Invalidation:** {t.invalidation}",
                ]
                if t.key_risks:
                    L.append("- **Risks:** " + "; ".join(t.key_risks))
                L.append("")
        if brief.watch_list:
            L += [
                "### Watch list (next few days)",
                "",
                "| When | What | Why | Tickers |",
                "|---|---|---|---|",
            ]
            for w in brief.watch_list:
                L.append(f"| {w.when} | {w.what} | {w.why_it_matters} | {', '.join(w.tickers)} |")
            L.append("")
        L += ["### Source reliability", "", brief.creator_reliability, ""]
        if brief.disagreements:
            L += ["### Disagreements", ""] + [f"- {d}" for d in brief.disagreements] + [""]
        if brief.risks_and_caveats:
            L += ["### Risks and caveats", ""] + [f"- {r}" for r in brief.risks_and_caveats] + [""]

    if stats:
        L += [
            "## Ticker mentions across videos",
            "",
            "| Ticker | Mentions | Bullish | Bearish | Net |",
            "|---|---|---|---|---|",
        ]
        for st in stats[:20]:
            L.append(
                f"| {st.ticker} | {st.mentions} | {st.bullish} | {st.bearish} | {st.net_score:+d} |"
            )
        L.append("")

    L += ["## Videos", ""]
    for i, v in enumerate(videos, 1):
        a = analyses.get(v.video_id)
        fc = fact_checks.get(v.video_id)
        L.append(f"### {i}. [{v.title}]({v.url})")
        meta = f"*{v.channel_title}*"
        if v.view_count:
            meta += f" · {_fmt_views(v.view_count)} views"
        if v.duration_seconds:
            meta += f" · {_duration(v.duration_seconds)}"
        if a:
            meta += (
                f" · {_ARROW.get(a.market_outlook, a.market_outlook)} · signal {a.signal_quality}/5"
            )
        if fc:
            meta += f" · reliability {fc.reliability_score}/5"
        L += [meta, ""]
        if a is None:
            L += [f"_Skipped: {(failures or {}).get(v.video_id, 'not analysed')}_", ""]
            continue
        L += ["**Summary.** " + a.summary, ""]
        if a.key_themes:
            L.append("**Themes:** " + ", ".join(a.key_themes))
        if a.macro_events:
            L.append("**Macro events:** " + ", ".join(a.macro_events))
        if a.tickers:
            L.append(
                "**Tickers:** "
                + ", ".join(
                    f"{m.ticker} ({m.sentiment}/{m.confidence}, {m.timeframe})"
                    for m in a.tickers[:10]
                )
            )
        if a.price_levels:
            L.append(
                "**Levels:** "
                + ", ".join(f"{p.ticker} {p.kind} {p.level}" for p in a.price_levels[:10])
            )
        if a.recommended_actions:
            L += ["", "**Speaker's recommendations:**"] + [
                f"- {r}" for r in a.recommended_actions[:6]
            ]
        if a.claims:
            L += ["", "**Key claims:**"]
            for c in a.claims:
                suffix = f" _(by {c.resolves_by})_" if c.resolves_by else ""
                tick = f" [{', '.join(c.tickers)}]" if c.tickers else ""
                L.append(f"- ({c.kind}) {c.text}{tick}{suffix}")
        if a.risk_flags:
            L += ["", "**Risk flags:** " + "; ".join(a.risk_flags)]
        if fc:
            L += ["", f"**Fact check ({fc.reliability_score}/5).** {fc.reliability_summary}", ""]
            if fc.verifications:
                L += ["| Claim | Verdict | Evidence |", "|---|---|---|"]
                for ver in fc.verifications:
                    src = " ".join(f"[{n}]({u})" for n, u in enumerate(ver.sources[:3], 1))
                    ev = ver.evidence.replace("|", "\\|")
                    claim = ver.claim.replace("|", "\\|")
                    verdict = _VERDICT.get(ver.verdict, ver.verdict)
                    L.append(f"| {claim} | {verdict} | {ev} {src} |")
                L.append("")
            if fc.corrections:
                L += ["**Corrections:**"] + [f"- {c}" for c in fc.corrections] + [""]
            if fc.new_context:
                L += (
                    ["**New context since recording:**"] + [f"- {c}" for c in fc.new_context] + [""]
                )
        elif fc_error := (failures or {}).get(f"factcheck:{v.video_id}"):
            L += ["", f"_Fact check failed: {fc_error}_", ""]
        L.append("")

    L += [
        "---",
        "_Generated automatically from public YouTube transcripts and web sources. Views are "
        "attributed to the creators. This is research tooling, not investment advice._",
    ]
    return "\n".join(L) + "\n"


def build_brief_json(
    target_date: date,
    videos: list[VideoMeta],
    analyses: dict[str, VideoAnalysis],
    fact_checks: dict[str, FactCheckReport],
    stats: list[TickerStats],
    brief: TradingBrief | None,
    *,
    failures: dict[str, str] | None = None,
    cost_usd: float = 0.0,
) -> dict:
    return {
        "date": target_date.isoformat(),
        "generated_cost_usd": round(cost_usd, 4),
        "brief": brief.model_dump() if brief else None,
        "ticker_stats": [s.model_dump() for s in stats],
        "videos": [
            {
                **v.model_dump(mode="json"),
                "url": v.url,
                "analysis": analyses[v.video_id].model_dump() if v.video_id in analyses else None,
                "fact_check": fact_checks[v.video_id].model_dump()
                if v.video_id in fact_checks
                else None,
                "failure": (failures or {}).get(v.video_id),
            }
            for v in videos
        ],
    }


def write_brief(reports_dir: Path, target_date: date, markdown: str, payload: dict) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"brief-{target_date.isoformat()}"
    md_path = reports_dir / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")
    (reports_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (reports_dir / "latest-brief.md").write_text(markdown, encoding="utf-8")
    return md_path
