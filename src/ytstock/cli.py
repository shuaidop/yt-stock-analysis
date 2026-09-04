"""``ytstock`` command-line interface."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import func, select

from ytstock import __version__
from ytstock.config import Settings, get_settings
from ytstock.db import Analysis, Database, PipelineRun, Transcript, Video
from ytstock.discovery import default_target_date
from ytstock.log import configure_logging, get_logger
from ytstock.pipeline import Pipeline

app = typer.Typer(
    help="Discover, transcribe, and analyse the day's popular stock-market YouTube videos.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
log = get_logger(__name__)

DateOpt = Annotated[
    str | None,
    typer.Option(
        "--date", "-d", help="Target trading day (YYYY-MM-DD). Default: today in market tz."
    ),
]


def _parse_date(value: str | None, settings: Settings) -> date:
    if value is None:
        return default_target_date(settings.market_timezone)
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter("expected YYYY-MM-DD") from exc


def _bootstrap() -> tuple[Settings, Pipeline]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    return settings, Pipeline(settings, Database(settings.database_url))


@app.callback(invoke_without_command=True)
def _root(
    version: Annotated[bool, typer.Option("--version", is_eager=True)] = False,
) -> None:
    if version:
        typer.echo(f"ytstock {__version__}")
        raise typer.Exit()


@app.command()
def run(
    date_: DateOpt = None,
    top: Annotated[int | None, typer.Option("--top", "-n", help="Override TOP_N_VIDEOS.")] = None,
    skip_discovery: Annotated[
        bool, typer.Option(help="Reuse videos already stored for the date.")
    ] = False,
) -> None:
    """Run every stage: discover -> transcribe -> analyze -> report."""
    settings, pipeline = _bootstrap()
    target = _parse_date(date_, settings)
    summary = pipeline.run_all(target, top_n=top, skip_discovery=skip_discovery)
    typer.echo(json.dumps(summary.model_dump(mode="json"), indent=2))
    if summary.discovered and not summary.analyzed and not _has_ok_analyses(pipeline, target):
        raise typer.Exit(code=2)


def _has_ok_analyses(pipeline: Pipeline, target: date) -> bool:
    return any(a.status == "ok" for a in pipeline.latest_analyses(target).values())


@app.command()
def discover(
    date_: DateOpt = None,
    top: Annotated[int | None, typer.Option("--top", "-n")] = None,
) -> None:
    """Find and store the day's top videos (YouTube Data API)."""
    settings, pipeline = _bootstrap()
    found = pipeline.discover(_parse_date(date_, settings), top_n=top)
    for v in found:
        mins = v.duration_seconds // 60
        typer.echo(f"{v.view_count:>9,}  {mins:>3}m  {v.channel_title[:24]:<24}  {v.title}")
    typer.echo(f"{len(found)} videos · quota used {pipeline.youtube.quota_used}")


@app.command()
def transcribe(
    date_: DateOpt = None,
    force: Annotated[bool, typer.Option(help="Re-fetch even if a transcript exists.")] = False,
) -> None:
    """Fetch transcripts for stored videos lacking one."""
    settings, pipeline = _bootstrap()
    ok, failed = pipeline.transcribe(_parse_date(date_, settings), force=force)
    typer.echo(f"transcripts ok={ok} failed={failed}")


@app.command()
def analyze(
    date_: DateOpt = None,
    force: Annotated[bool, typer.Option(help="Re-analyse even if a result exists.")] = False,
) -> None:
    """Run Claude extraction on every transcribed video."""
    settings, pipeline = _bootstrap()
    ok, failed = pipeline.analyze(_parse_date(date_, settings), force=force)
    typer.echo(f"analyses ok={ok} failed={failed}")


@app.command()
def report(
    date_: DateOpt = None,
    no_synthesis: Annotated[
        bool, typer.Option("--no-synthesis", help="Skip the cross-video Claude synthesis.")
    ] = False,
) -> None:
    """Write the Markdown + JSON report for the day."""
    settings, pipeline = _bootstrap()
    path, cost = pipeline.report(_parse_date(date_, settings), synthesize=not no_synthesis)
    typer.echo(f"wrote {path} (day cost ${cost:.2f})")


@app.command()
def brief(
    urls: Annotated[list[str], typer.Argument(help="YouTube URLs or 11-char video ids.")],
    date_: DateOpt = None,
    no_factcheck: Annotated[
        bool, typer.Option("--no-factcheck", help="Skip web-search fact-checking.")
    ] = False,
    include_daily: Annotated[
        bool, typer.Option(help="Also include videos already discovered for the date.")
    ] = False,
    no_portfolio: Annotated[
        bool, typer.Option("--no-portfolio", help="Do not feed portfolio/quotes into the brief.")
    ] = False,
) -> None:
    """Analyse specific videos end-to-end and write a fact-checked trading brief."""
    settings, pipeline = _bootstrap()
    if no_portfolio:
        settings.brief_include_portfolio = False
        settings.quotes_enabled = False
    target = _parse_date(date_, settings)
    summary = pipeline.run_brief(
        urls, target, fact_check=not no_factcheck, include_daily=include_daily
    )
    typer.echo(json.dumps(summary.model_dump(mode="json"), indent=2))
    if summary.analyzed == 0 and not _has_ok_analyses(pipeline, target):
        raise typer.Exit(code=2)


@app.command()
def portfolio() -> None:
    """Show the portfolio the brief will use (TWS if enabled, else the JSON snapshot)."""
    from ytstock.portfolio import load_portfolio, portfolio_context

    settings, _ = _bootstrap()
    p = load_portfolio(settings)
    if p is None:
        typer.echo(
            f"no portfolio: enable IBKR_ENABLED or create {settings.portfolio_file}", err=True
        )
        raise typer.Exit(code=1)
    typer.echo(portfolio_context(p))


@app.command("portfolio-import")
def portfolio_import(
    summary: Annotated[Path, typer.Argument(help="JSON from get_account_summary")],
    positions: Annotated[Path, typer.Argument(help="JSON from get_account_positions")],
    orders: Annotated[Path | None, typer.Argument(help="JSON from get_account_orders")] = None,
) -> None:
    """Convert IBKR-connector JSON dumps into the portfolio snapshot file."""
    from ytstock.portfolio import portfolio_from_ibkr_connector, save_portfolio_file

    settings, _ = _bootstrap()
    p = portfolio_from_ibkr_connector(
        json.loads(summary.read_text()),
        json.loads(positions.read_text()),
        json.loads(orders.read_text()) if orders else None,
    )
    save_portfolio_file(settings.portfolio_file, p)
    typer.echo(
        f"wrote {settings.portfolio_file}: {len(p.holdings)} holdings, {len(p.orders)} orders"
    )


@app.command()
def quotes(
    symbols: Annotated[list[str], typer.Argument(help="Tickers, e.g. SPY AVGO ^VIX")],
) -> None:
    """Fetch current quotes (yfinance)."""
    from ytstock.portfolio import fetch_quotes, quotes_context

    _bootstrap()
    typer.echo(quotes_context(fetch_quotes(symbols)) or "no quotes")


@app.command("brief-render")
def brief_render(date_: DateOpt = None) -> None:
    """Re-render the day's brief from stored results (no model calls)."""
    settings, pipeline = _bootstrap()
    path, _ = pipeline.brief(_parse_date(date_, settings), regenerate=False)
    typer.echo(f"wrote {path}")


@app.command()
def factcheck(
    date_: DateOpt = None,
    force: Annotated[bool, typer.Option(help="Re-check even if a result exists.")] = False,
) -> None:
    """Fact-check the claims of every analysed video for the day (uses web search)."""
    settings, pipeline = _bootstrap()
    ok, failed = pipeline.fact_check(_parse_date(date_, settings), force=force)
    typer.echo(f"fact checks ok={ok} failed={failed}")


@app.command()
def status(date_: DateOpt = None) -> None:
    """Show per-stage counts for the day."""
    settings, pipeline = _bootstrap()
    target = _parse_date(date_, settings)
    with pipeline.db.session() as s:
        videos = s.scalar(
            select(func.count()).select_from(Video).where(Video.target_date == target)
        )
        transcripts: dict[str, int] = dict(
            s.execute(
                select(Transcript.status, func.count())
                .join(Video)
                .where(Video.target_date == target)
                .group_by(Transcript.status)
            )
            .tuples()
            .all()
        )
        analyses: dict[str, int] = dict(
            s.execute(
                select(Analysis.status, func.count())
                .join(Video)
                .where(Video.target_date == target)
                .group_by(Analysis.status)
            )
            .tuples()
            .all()
        )
        cost = s.scalar(
            select(func.coalesce(func.sum(Analysis.cost_usd), 0.0))
            .join(Video)
            .where(Video.target_date == target)
        )
        runs = s.scalars(
            select(PipelineRun)
            .where(PipelineRun.target_date == target)
            .order_by(PipelineRun.started_at.desc())
            .limit(8)
        ).all()
    typer.echo(f"date: {target}")
    typer.echo(f"videos: {videos}")
    typer.echo(f"transcripts: {transcripts or {}}")
    typer.echo(f"analyses: {analyses or {}}  (cost ${cost:.2f})")
    typer.echo("recent stage runs:")
    for r in runs:
        typer.echo(f"  {r.started_at:%H:%M:%S}  {r.stage:<10} {r.status:<7} {r.detail or ''}")


@app.command()
def prune(
    transcript_days: Annotated[
        int, typer.Option(help="Drop transcript text for videos older than N days (0 = skip).")
    ] = 30,
    json_days: Annotated[
        int, typer.Option(help="Delete per-day report JSON older than N days (0 = skip).")
    ] = 0,
) -> None:
    """Free disk: old transcript text, old report JSON, leftover Whisper temp dirs."""
    _, pipeline = _bootstrap()
    out = pipeline.prune(transcript_days=transcript_days, json_days=json_days)
    typer.echo(json.dumps(out))


@app.command("init-db")
def init_db_cmd() -> None:
    """Create tables (safe to run repeatedly)."""
    settings, _pipeline = _bootstrap()
    typer.echo(f"database ready: {settings.database_url}")


@app.command()
def show(video_id: str) -> None:
    """Print the stored analysis for one video."""
    _, pipeline = _bootstrap()
    with pipeline.db.session() as s:
        row = s.scalars(
            select(Analysis)
            .where(Analysis.video_id == video_id)
            .order_by(Analysis.created_at.desc())
        ).first()
    if row is None:
        typer.echo("no analysis found", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        json.dumps({"status": row.status, "error": row.error, "result": row.result}, indent=2)
    )


@app.command()
def sources() -> None:
    """Print the resolved sources file."""
    settings = get_settings()
    typer.echo(Path(settings.sources_file).read_text())


if __name__ == "__main__":  # pragma: no cover
    app()
