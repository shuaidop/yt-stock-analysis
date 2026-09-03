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
_ARROW_ZH = {"bullish": "▲ 看多", "bearish": "▼ 看空", "neutral": "— 中性", "mixed": "◆ 分歧"}

# Fixed headings/labels around the model-written prose (which follows REPORT_LANGUAGE).
_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "digest_title": "Stock-market YouTube digest",
        "brief_title": "Trading brief",
        "analysed": "{a} of {n} videos analysed",
        "fact_checked": "{k} fact-checked",
        "combined_views": "{v:,} combined views",
        "est_cost": "est. cost ${c:.2f}",
        "consensus": "Consensus",
        "agreement": "{level} agreement",
        "ticker_consensus": "Ticker consensus",
        "contrarian": "Contrarian views",
        "catalysts": "Upcoming catalysts",
        "predictions": "Predictions to track",
        "ticker_mentions": "Ticker mentions",
        "ticker_mentions_across": "Ticker mentions across videos",
        "videos": "Videos",
        "views": "views",
        "signal": "signal {q}/5",
        "reliability": "reliability {r}/5",
        "skipped": "Skipped: {reason}",
        "summary": "Summary.",
        "themes": "Themes",
        "macro_events": "Macro events",
        "tickers": "Tickers",
        "levels": "Levels",
        "recommendations": "Speaker's recommendations",
        "key_claims": "Key claims",
        "risk_flags": "Risk flags",
        "fact_check": "Fact check ({r}/5).",
        "fc_table": "| Claim | Verdict | Evidence |",
        "corrections": "Corrections",
        "new_context": "New context since recording",
        "fc_failed": "Fact check failed: {e}",
        "macro": "Macro / economy",
        "equity": "Equity market",
        "sectors": "Sectors and single names",
        "derivatives": "Derivatives, rates and flows",
        "hidden": "Hidden logic / second-order ideas",
        "trade_ideas": "Trade ideas",
        "ti_table": "| # | Idea | Ticker | Instrument | Dir | Horizon | Conviction |",
        "basis": "Basis",
        "catalyst": "Catalyst",
        "entry": "Entry conditions",
        "invalidation": "Invalidation",
        "risks": "Risks",
        "watch": "Watch list (next few days)",
        "watch_table": "| When | What | Why | Tickers |",
        "source_reliability": "Source reliability",
        "disagreements": "Disagreements",
        "risks_caveats": "Risks and caveats",
        "mention_table": "| Ticker | Mentions | Bullish | Bearish | Neutral/Mixed | Net |",
        "mention_table_brief": "| Ticker | Mentions | Bullish | Bearish | Net |",
        "footer_digest": "Generated automatically from public YouTube transcripts. Views are "
        "attributed to the creators and are not investment advice.",
        "footer_brief": "Generated automatically from public YouTube transcripts and web sources. "
        "Views are attributed to the creators. This is research tooling, not investment advice.",
    },
    "zh": {
        "digest_title": "美股 YouTube 每日摘要",
        "brief_title": "交易简报",
        "analysed": "{n} 个视频中 {a} 个完成分析",
        "fact_checked": "{k} 个完成事实核查",
        "combined_views": "合计播放 {v:,}",
        "est_cost": "估算成本 ${c:.2f}",
        "consensus": "共识",
        "agreement": "一致程度：{level}",
        "ticker_consensus": "标的共识",
        "contrarian": "反向观点",
        "catalysts": "即将到来的催化剂",
        "predictions": "待跟踪的预测",
        "ticker_mentions": "标的提及统计",
        "ticker_mentions_across": "跨视频标的提及统计",
        "videos": "视频",
        "views": "次播放",
        "signal": "信号质量 {q}/5",
        "reliability": "可信度 {r}/5",
        "skipped": "跳过：{reason}",
        "summary": "摘要。",
        "themes": "主题",
        "macro_events": "宏观事件",
        "tickers": "标的",
        "levels": "关键价位",
        "recommendations": "博主的操作建议",
        "key_claims": "关键论断",
        "risk_flags": "风险提示",
        "fact_check": "事实核查（{r}/5）。",
        "fc_table": "| 论断 | 结论 | 证据 |",
        "corrections": "更正",
        "new_context": "录制后的新情况",
        "fc_failed": "事实核查失败：{e}",
        "macro": "宏观与经济",
        "equity": "股票市场",
        "sectors": "板块与个股",
        "derivatives": "衍生品、利率与资金流",
        "hidden": "隐藏逻辑 / 二阶观点",
        "trade_ideas": "交易想法",
        "ti_table": "| # | 想法 | 标的 | 工具 | 方向 | 周期 | 信心 |",
        "basis": "依据",
        "catalyst": "催化剂",
        "entry": "进场条件",
        "invalidation": "失效条件",
        "risks": "风险",
        "watch": "观察清单（未来几天）",
        "watch_table": "| 时间 | 事项 | 为什么重要 | 标的 |",
        "source_reliability": "来源可信度",
        "disagreements": "分歧",
        "risks_caveats": "风险与注意事项",
        "mention_table": "| 标的 | 提及 | 看多 | 看空 | 中性/分歧 | 净值 |",
        "mention_table_brief": "| 标的 | 提及 | 看多 | 看空 | 净值 |",
        "footer_digest": "本文由公开 YouTube 字幕自动生成，观点归属于各博主，不构成投资建议。",
        "footer_brief": "本文由公开 YouTube 字幕和网络来源自动生成，观点归属于各博主。"
        "这是研究工具，不构成投资建议。",
    },
}


def _lang_key(report_language: str) -> str:
    lang = report_language.lower()
    if "chinese" in lang or "中文" in lang or lang.startswith("zh"):
        return "zh"
    return "en"


def _t(lang: str) -> dict[str, str]:
    return _STRINGS[_lang_key(lang)]


def _arrow(lang: str, sentiment: str) -> str:
    table = _ARROW_ZH if _lang_key(lang) == "zh" else _ARROW
    return table.get(sentiment, sentiment)


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
    language: str = "English",
) -> str:
    T = _t(language)
    ar = lambda x: _arrow(language, x)  # noqa: E731
    lines: list[str] = [f"# {T['digest_title']} — {target_date.isoformat()}", ""]
    analysed = [v for v in videos if v.video_id in analyses]
    lines.append(
        T["analysed"].format(a=len(analysed), n=len(videos))
        + " · "
        + T["combined_views"].format(v=sum(v.view_count for v in analysed))
        + " · "
        + T["est_cost"].format(c=cost_usd)
    )
    lines.append("")

    if synthesis:
        lines += [
            f"## {synthesis.headline}",
            "",
            f"**{T['consensus']}:** {ar(synthesis.consensus_outlook)}"
            f" ({T['agreement'].format(level=synthesis.agreement_level)})",
            "",
            synthesis.market_narrative,
            "",
        ]
        if synthesis.top_tickers:
            lines += [
                f"### {T['ticker_consensus']}",
                "",
                "| Ticker | View | Note |",
                "|---|---|---|",
            ]
            for t in synthesis.top_tickers:
                lines.append(f"| **{t.ticker}** | {ar(t.consensus)} | {t.note} |")
            lines.append("")
        for key, items in (
            ("contrarian", synthesis.contrarian_views),
            ("catalysts", synthesis.upcoming_catalysts),
            ("predictions", synthesis.predictions_to_track),
        ):
            if items:
                lines += [f"### {T[key]}", ""] + [f"- {c}" for c in items] + [""]

    if stats:
        lines += [f"## {T['ticker_mentions']}", "", T["mention_table"], "|---|---|---|---|---|---|"]
        for st in stats[:20]:
            lines.append(
                f"| {st.ticker} | {st.mentions} | {st.bullish} | {st.bearish} | "
                f"{st.neutral + st.mixed} | {st.net_score:+d} |"
            )
        lines.append("")

    lines += [f"## {T['videos']}", ""]
    for i, v in enumerate(videos, 1):
        a = analyses.get(v.video_id)
        lines.append(f"### {i}. [{v.title}]({v.url})")
        meta = (
            f"*{v.channel_title}* · {_fmt_views(v.view_count)} {T['views']} · "
            f"{_duration(v.duration_seconds)}"
        )
        if a:
            meta += f" · {ar(a.market_outlook)} · {T['signal'].format(q=a.signal_quality)}"
        lines += [meta, ""]
        if a is None:
            reason = (failures or {}).get(v.video_id, "not analysed")
            lines += [f"_{T['skipped'].format(reason=reason)}_", ""]
            continue
        lines += [a.summary, ""]
        if a.key_themes:
            lines.append(f"**{T['themes']}:** " + ", ".join(a.key_themes))
        if a.tickers:
            primary = [m for m in a.tickers if m.is_primary] or a.tickers
            lines.append(
                f"**{T['tickers']}:** "
                + ", ".join(f"{m.ticker} ({m.sentiment}, {m.timeframe})" for m in primary[:8])
            )
        if a.claims:
            lines += ["", f"**{T['key_claims']}:**"]
            for c in a.claims[:5]:
                suffix = f" _({c.resolves_by})_" if c.resolves_by else ""
                tick = f" [{', '.join(c.tickers)}]" if c.tickers else ""
                lines.append(f"- ({c.kind}) {c.text}{tick}{suffix}")
        if a.risk_flags:
            lines += ["", f"**{T['risk_flags']}:** " + "; ".join(a.risk_flags)]
        lines.append("")

    lines += ["---", f"_{T['footer_digest']}_"]
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


def report_dir(reports_dir: Path, target_date: date) -> Path:
    """reports/<YYYY>/<YYYY-MM-DD>/ - one folder per trading day."""
    return reports_dir / f"{target_date:%Y}" / target_date.isoformat()


def write_report(reports_dir: Path, target_date: date, markdown: str, payload: dict) -> Path:
    folder = report_dir(reports_dir, target_date)
    folder.mkdir(parents=True, exist_ok=True)
    md_path = folder / "digest.md"
    md_path.write_text(markdown, encoding="utf-8")
    (folder / "digest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (reports_dir / "latest-digest.md").write_text(markdown, encoding="utf-8")
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
    language: str = "English",
) -> str:
    T = _t(language)
    ar = lambda x: _arrow(language, x)  # noqa: E731
    L: list[str] = [f"# {T['brief_title']} — {target_date.isoformat()}", ""]
    L.append(
        T["analysed"].format(a=len(analyses), n=len(videos))
        + " · "
        + T["fact_checked"].format(k=len(fact_checks))
        + " · "
        + T["est_cost"].format(c=cost_usd)
    )
    L.append("")

    if brief:
        L += [f"## {brief.headline}", "", brief.executive_summary, ""]
        L += [f"### {T['macro']}", "", brief.macro_economy, ""]
        L += [f"### {T['equity']}", "", brief.equity_market, ""]
        L += [f"### {T['sectors']}", "", brief.sectors_and_stocks, ""]
        L += [f"### {T['derivatives']}", "", brief.derivatives_and_flows, ""]
        if brief.hidden_logic:
            L += [f"### {T['hidden']}", ""] + [f"- {h}" for h in brief.hidden_logic] + [""]
        if brief.trade_ideas:
            L += [f"### {T['trade_ideas']}", "", T["ti_table"], "|---|---|---|---|---|---|---|"]
            for i, t in enumerate(brief.trade_ideas, 1):
                L.append(
                    f"| {i} | {t.title} | {t.ticker} | {t.instrument} | {t.direction} | "
                    f"{t.horizon} | {t.conviction} |"
                )
            L.append("")
            for i, t in enumerate(brief.trade_ideas, 1):
                L += [
                    f"#### {i}. {t.title} ({t.ticker}, {t.direction} / {t.instrument})",
                    "",
                    t.thesis,
                    "",
                    f"- **{T['basis']}:** {t.source_basis}",
                    f"- **{T['catalyst']}:** {t.catalyst}",
                    f"- **{T['entry']}:** {t.entry_conditions}",
                    f"- **{T['invalidation']}:** {t.invalidation}",
                ]
                if t.key_risks:
                    L.append(f"- **{T['risks']}:** " + "; ".join(t.key_risks))
                L.append("")
        if brief.watch_list:
            L += [f"### {T['watch']}", "", T["watch_table"], "|---|---|---|---|"]
            for w in brief.watch_list:
                L.append(f"| {w.when} | {w.what} | {w.why_it_matters} | {', '.join(w.tickers)} |")
            L.append("")
        L += [f"### {T['source_reliability']}", "", brief.creator_reliability, ""]
        if brief.disagreements:
            L += [f"### {T['disagreements']}", ""] + [f"- {d}" for d in brief.disagreements] + [""]
        if brief.risks_and_caveats:
            L += (
                [f"### {T['risks_caveats']}", ""]
                + [f"- {r}" for r in brief.risks_and_caveats]
                + [""]
            )

    if stats:
        L += [
            f"## {T['ticker_mentions_across']}",
            "",
            T["mention_table_brief"],
            "|---|---|---|---|---|",
        ]
        for st in stats[:20]:
            L.append(
                f"| {st.ticker} | {st.mentions} | {st.bullish} | {st.bearish} | {st.net_score:+d} |"
            )
        L.append("")

    L += [f"## {T['videos']}", ""]
    for i, v in enumerate(videos, 1):
        a = analyses.get(v.video_id)
        fc = fact_checks.get(v.video_id)
        L.append(f"### {i}. [{v.title}]({v.url})")
        meta = f"*{v.channel_title}*"
        if v.view_count:
            meta += f" · {_fmt_views(v.view_count)} {T['views']}"
        if v.duration_seconds:
            meta += f" · {_duration(v.duration_seconds)}"
        if a:
            meta += f" · {ar(a.market_outlook)} · {T['signal'].format(q=a.signal_quality)}"
        if fc:
            meta += f" · {T['reliability'].format(r=fc.reliability_score)}"
        L += [meta, ""]
        if a is None:
            reason = (failures or {}).get(v.video_id, "not analysed")
            L += [f"_{T['skipped'].format(reason=reason)}_", ""]
            continue
        L += [f"**{T['summary']}** " + a.summary, ""]
        if a.key_themes:
            L.append(f"**{T['themes']}:** " + ", ".join(a.key_themes))
        if a.macro_events:
            L.append(f"**{T['macro_events']}:** " + ", ".join(a.macro_events))
        if a.tickers:
            L.append(
                f"**{T['tickers']}:** "
                + ", ".join(
                    f"{m.ticker} ({m.sentiment}/{m.confidence}, {m.timeframe})"
                    for m in a.tickers[:10]
                )
            )
        if a.price_levels:
            L.append(
                f"**{T['levels']}:** "
                + ", ".join(f"{p.ticker} {p.kind} {p.level}" for p in a.price_levels[:10])
            )
        if a.recommended_actions:
            L += ["", f"**{T['recommendations']}:**"] + [
                f"- {r}" for r in a.recommended_actions[:6]
            ]
        if a.claims:
            L += ["", f"**{T['key_claims']}:**"]
            for c in a.claims:
                suffix = f" _({c.resolves_by})_" if c.resolves_by else ""
                tick = f" [{', '.join(c.tickers)}]" if c.tickers else ""
                L.append(f"- ({c.kind}) {c.text}{tick}{suffix}")
        if a.risk_flags:
            L += ["", f"**{T['risk_flags']}:** " + "; ".join(a.risk_flags)]
        if fc:
            L += [
                "",
                f"**{T['fact_check'].format(r=fc.reliability_score)}** {fc.reliability_summary}",
                "",
            ]
            if fc.verifications:
                L += [T["fc_table"], "|---|---|---|"]
                for ver in fc.verifications:
                    src = " ".join(f"[{n}]({u})" for n, u in enumerate(ver.sources[:3], 1))
                    ev = ver.evidence.replace("|", "\\|")
                    claim = ver.claim.replace("|", "\\|")
                    verdict = _VERDICT.get(ver.verdict, ver.verdict)
                    L.append(f"| {claim} | {verdict} | {ev} {src} |")
                L.append("")
            if fc.corrections:
                L += [f"**{T['corrections']}:**"] + [f"- {c}" for c in fc.corrections] + [""]
            if fc.new_context:
                L += [f"**{T['new_context']}:**"] + [f"- {c}" for c in fc.new_context] + [""]
        elif fc_error := (failures or {}).get(f"factcheck:{v.video_id}"):
            L += ["", f"_{T['fc_failed'].format(e=fc_error)}_", ""]
        L.append("")

    L += ["---", f"_{T['footer_brief']}_"]
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
    folder = report_dir(reports_dir, target_date)
    folder.mkdir(parents=True, exist_ok=True)
    md_path = folder / "brief.md"
    md_path.write_text(markdown, encoding="utf-8")
    (folder / "brief.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (reports_dir / "latest-brief.md").write_text(markdown, encoding="utf-8")
    return md_path
