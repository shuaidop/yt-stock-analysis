"""Stage orchestration. Each stage is idempotent and resumable.

    discover  -> upsert Video rows for the target date (refreshes view counts)
    transcribe -> Transcript rows for videos lacking a successful one
    analyze   -> Analysis rows (per PROMPT_VERSION) for videos with transcripts
    report    -> DailySynthesis + Markdown/JSON files + DailyReport row

``run_all`` chains them. A failure in one video never aborts the day; failures are
recorded and surfaced in the report and the ``RunSummary``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select, text, true, update

from ytstock.analysis import (
    BRIEF_PROMPT_VERSION,
    FACTCHECK_PROMPT_VERSION,
    PROMPT_VERSION,
    GroundedAnalyzer,
    compute_ticker_stats,
)
from ytstock.config import Settings
from ytstock.db import (
    Analysis,
    Brief,
    DailyReport,
    Database,
    FactCheck,
    PipelineRun,
    Transcript,
    Video,
    utcnow,
)
from ytstock.discovery import DiscoveryWindow, Sources, discover
from ytstock.log import get_logger
from ytstock.report import (
    build_brief_json,
    build_json,
    render_brief_markdown,
    render_markdown,
    write_brief,
    write_report,
)
from ytstock.schemas import (
    DailySynthesis,
    FactCheckReport,
    RunSummary,
    TradingBrief,
    VideoAnalysis,
    VideoMeta,
)
from ytstock.transcripts import TranscriptService, TranscriptTransientError, TranscriptUnavailable
from ytstock.youtube import YouTubeClient, extract_video_id, fetch_metadata_for_ids
from ytstock.ytdlp_discovery import YtDlpClient

log = get_logger(__name__)

MAX_TRANSCRIPT_ATTEMPTS = 3


def _video_meta(v: Video) -> VideoMeta:
    return VideoMeta(
        video_id=v.video_id,
        title=v.title,
        channel_id=v.channel_id,
        channel_title=v.channel_title,
        published_at=v.published_at,
        duration_seconds=v.duration_seconds,
        view_count=v.view_count,
        like_count=v.like_count,
        comment_count=v.comment_count,
        description=v.description,
        discovery_source=v.discovery_source,
    )


class _StageRun:
    """Context manager that records a PipelineRun row."""

    def __init__(self, db: Database, target_date: date, stage: str) -> None:
        self._db, self._date, self._stage = db, target_date, stage
        self.detail: dict[str, Any] = {}
        self._id: int | None = None

    def __enter__(self) -> _StageRun:
        with self._db.session() as s:
            row = PipelineRun(target_date=self._date, stage=self._stage, status="running")
            s.add(row)
            s.flush()
            self._id = row.id
        log.info("stage.start", stage=self._stage, date=self._date.isoformat())
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        status = "error" if exc_type else "ok"
        if exc is not None:
            self.detail["error"] = str(exc)
        with self._db.session() as s:
            row = s.get(PipelineRun, self._id)
            if row is not None:
                row.status = status
                row.detail = self.detail
                row.finished_at = utcnow()
        log.info("stage.end", stage=self._stage, status=status, detail=self.detail)


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        *,
        youtube: Any | None = None,
        transcripts: TranscriptService | None = None,
        analyzer: GroundedAnalyzer | None = None,
        sources: Sources | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self._youtube = youtube
        self._transcripts = transcripts
        self._analyzer = analyzer
        self._sources = sources

    # ---- lazy deps (so `report` never needs API keys) ----------------------
    @property
    def youtube(self) -> Any:
        """YouTube Data API client, or the keyless yt-dlp client (same interface)."""
        if self._youtube is None:
            s = self.settings
            backend = s.youtube_backend
            if backend == "auto":
                backend = "api" if s.youtube_api_key else "ytdlp"
            log.info("youtube.backend", backend=backend)
            if backend == "api":
                self._youtube = YouTubeClient(s.youtube_api_key)
            else:
                self._youtube = YtDlpClient(
                    region_code=s.region_code,
                    proxy_url=s.yt_proxy_url,
                    min_duration=s.min_duration_seconds,
                    max_duration=s.max_duration_seconds,
                    min_views=s.min_view_count,
                    exclude_keywords=self.sources.exclude_keywords,
                    hydrate_workers=s.ytdlp_hydrate_workers,
                )
        return self._youtube

    @property
    def transcripts(self) -> TranscriptService:
        if self._transcripts is None:
            self._transcripts = TranscriptService.from_settings(self.settings)
        return self._transcripts

    @property
    def analyzer(self) -> GroundedAnalyzer:
        if self._analyzer is None:
            self._analyzer = GroundedAnalyzer(self.settings)
        return self._analyzer

    @property
    def sources(self) -> Sources:
        if self._sources is None:
            self._sources = Sources.load(self.settings.sources_file)
        return self._sources

    # ---- queries -----------------------------------------------------------
    def videos_for(self, target_date: date) -> list[Video]:
        with self.db.session() as s:
            rows = s.scalars(
                select(Video)
                .where(Video.target_date == target_date)
                .order_by(Video.view_count.desc())
            ).all()
        return list(rows)

    def latest_analyses(self, target_date: date) -> dict[str, Analysis]:
        with self.db.session() as s:
            rows = s.scalars(
                select(Analysis)
                .join(Video)
                .where(Video.target_date == target_date, Analysis.prompt_version == PROMPT_VERSION)
                .order_by(Analysis.created_at)
            ).all()
        latest: dict[str, Analysis] = {}
        for a in rows:  # later rows overwrite earlier -> latest attempt wins
            latest[a.video_id] = a
        return latest

    # ---- stages ------------------------------------------------------------
    def discover(self, target_date: date, *, top_n: int | None = None) -> list[VideoMeta]:
        window = DiscoveryWindow.for_date(target_date, self.settings.market_timezone)
        with _StageRun(self.db, target_date, "discover") as run:
            found = discover(
                self.youtube,
                settings=self.settings,
                sources=self.sources,
                window=window,
                top_n=top_n,
            )
            with self.db.session() as s:
                for meta in found:
                    row = s.get(Video, meta.video_id)
                    if row is None:
                        row = Video(
                            video_id=meta.video_id, target_date=target_date, discovery_source=""
                        )
                        s.add(row)
                    for field in (
                        "title",
                        "channel_id",
                        "channel_title",
                        "published_at",
                        "duration_seconds",
                        "view_count",
                        "like_count",
                        "comment_count",
                        "description",
                    ):
                        setattr(row, field, getattr(meta, field))
                    if meta.discovery_source and not row.discovery_source.startswith("channel:"):
                        row.discovery_source = meta.discovery_source
                    row.stats_updated_at = utcnow()
            run.detail.update(found=len(found), quota_used=self.youtube.quota_used)
        return found

    def transcribe(
        self,
        target_date: date,
        *,
        video_ids: list[str] | None = None,
        force: bool = False,
    ) -> tuple[int, int]:
        """Returns (succeeded, failed) for this invocation."""
        ok = failed = 0
        with _StageRun(self.db, target_date, "transcribe") as run:
            with self.db.session() as s:
                pending = s.scalars(
                    select(Video)
                    .outerjoin(Transcript)
                    .where(
                        Video.target_date == target_date,
                        true()
                        if force
                        else (
                            (Transcript.video_id.is_(None))
                            | (
                                (Transcript.status != "ok")
                                & (Transcript.attempts < MAX_TRANSCRIPT_ATTEMPTS)
                            )
                        ),
                    )
                ).all()
                pending_ids = [
                    v.video_id for v in pending if video_ids is None or v.video_id in video_ids
                ]

            for video_id in pending_ids:
                status, payload = self._fetch_one_transcript(video_id)
                with self.db.session() as s:
                    row = s.get(Transcript, video_id)
                    if row is None:
                        row = Transcript(video_id=video_id, status=status, attempts=0)
                        s.add(row)
                    row.status = status
                    row.attempts += 1
                    row.fetched_at = utcnow()
                    row.error = payload.get("error", "")
                    if status == "ok":
                        row.text = payload["text"]
                        row.language = payload["language"]
                        row.source = payload["source"]
                        row.is_generated = payload["is_generated"]
                        row.char_count = len(payload["text"])
                if status == "ok":
                    ok += 1
                else:
                    failed += 1
            run.detail.update(pending=len(pending_ids), ok=ok, failed=failed)
        return ok, failed

    def _fetch_one_transcript(self, video_id: str) -> tuple[str, dict[str, Any]]:
        try:
            result = self.transcripts.fetch(video_id)
        except TranscriptUnavailable as exc:
            # permanent: mark with max attempts so we stop retrying
            return "unavailable", {"error": str(exc)}
        except TranscriptTransientError as exc:
            return "error", {"error": str(exc)}
        except Exception as exc:
            log.exception("transcript.unexpected", video_id=video_id)
            return "error", {"error": f"{exc.__class__.__name__}: {exc}"}
        return "ok", {
            "text": result.text,
            "language": result.language,
            "source": result.source,
            "is_generated": result.is_generated,
        }

    def analyze(
        self, target_date: date, *, force: bool = False, video_ids: list[str] | None = None
    ) -> tuple[int, int]:
        ok = failed = 0
        with _StageRun(self.db, target_date, "analyze") as run:
            done = (
                set()
                if force
                else {
                    vid for vid, a in self.latest_analyses(target_date).items() if a.status == "ok"
                }
            )
            with self.db.session() as s:
                rows = s.execute(
                    select(Video, Transcript)
                    .join(Transcript)
                    .where(Video.target_date == target_date, Transcript.status == "ok")
                ).all()
                work = [
                    (_video_meta(v), t.text)
                    for v, t in rows
                    if v.video_id not in done and (video_ids is None or v.video_id in video_ids)
                ]

            def _one(item: tuple[VideoMeta, str]) -> tuple[VideoMeta, Any]:
                meta, text = item
                return meta, self.analyzer.analyze_video(meta, text)

            workers = max(1, self.settings.analysis_concurrency)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_one, item) for item in work]
                for fut in as_completed(futures):
                    meta, outcome = fut.result()
                    with self.db.session() as s:
                        s.add(
                            Analysis(
                                video_id=meta.video_id,
                                prompt_version=PROMPT_VERSION,
                                model=outcome.usage.model,
                                status=outcome.status,
                                result=outcome.result.model_dump() if outcome.result else None,
                                error=outcome.error,
                                input_tokens=outcome.usage.input_tokens,
                                output_tokens=outcome.usage.output_tokens,
                                cache_read_input_tokens=outcome.usage.cache_read_input_tokens,
                                cache_creation_input_tokens=outcome.usage.cache_creation_input_tokens,
                                cost_usd=outcome.usage.cost_usd,
                                request_id=outcome.request_id,
                            )
                        )
                    if outcome.status == "ok":
                        ok += 1
                    else:
                        failed += 1
                        log.warning(
                            "analysis.video.failed",
                            video_id=meta.video_id,
                            status=outcome.status,
                            error=outcome.error[:300],
                        )
            run.detail.update(pending=len(work), ok=ok, failed=failed)
        return ok, failed

    def report(self, target_date: date, *, synthesize: bool = True) -> tuple[str, float]:
        """Build the daily report. Returns (markdown_path, day_cost_usd)."""
        with _StageRun(self.db, target_date, "report") as run:
            videos = self.videos_for(target_date)
            metas = [_video_meta(v) for v in videos]
            analyses_rows = self.latest_analyses(target_date)
            analyses: dict[str, VideoAnalysis] = {}
            failures: dict[str, str] = {}
            day_cost = 0.0
            with self.db.session() as s:
                for v in videos:
                    a = analyses_rows.get(v.video_id)
                    if a is not None and a.status == "ok" and a.result:
                        analyses[v.video_id] = VideoAnalysis.model_validate(a.result)
                    elif a is not None:
                        failures[v.video_id] = f"analysis {a.status}: {a.error[:160]}"
                    else:
                        t = s.get(Transcript, v.video_id)
                        failures[v.video_id] = (
                            f"transcript {t.status}: {t.error[:160]}" if t else "no transcript yet"
                        )
                    for a_row in s.scalars(select(Analysis).where(Analysis.video_id == v.video_id)):
                        day_cost += a_row.cost_usd

            stats = compute_ticker_stats(analyses)
            synthesis: DailySynthesis | None = None
            if synthesize and analyses:
                outcome = self.analyzer.synthesize_day(
                    target_date.isoformat(), metas, analyses, stats
                )
                if outcome.status == "ok":
                    synthesis = outcome.result
                else:
                    log.warning("synthesis.failed", status=outcome.status, error=outcome.error)
                day_cost += outcome.usage.cost_usd

            markdown = render_markdown(
                target_date,
                metas,
                analyses,
                stats,
                synthesis,
                failures=failures,
                cost_usd=day_cost,
                language=self.settings.report_language,
            )
            payload = build_json(
                target_date, metas, analyses, stats, synthesis, failures=failures, cost_usd=day_cost
            )
            path = write_report(self.settings.reports_dir, target_date, markdown, payload)

            with self.db.session() as s:
                row = s.get(DailyReport, target_date)
                if row is None:
                    row = DailyReport(
                        target_date=target_date,
                        prompt_version=PROMPT_VERSION,
                        model=self.settings.claude_model,
                        video_count=0,
                    )
                    s.add(row)
                row.prompt_version = PROMPT_VERSION
                row.model = self.settings.claude_model
                row.video_count = len(analyses)
                row.synthesis = synthesis.model_dump() if synthesis else None
                row.ticker_stats = [st.model_dump() for st in stats]
                row.markdown = markdown
                row.cost_usd = day_cost
                row.created_at = utcnow()
            run.detail.update(
                videos=len(videos), analysed=len(analyses), cost_usd=round(day_cost, 4)
            )
        return str(path), day_cost

    # ---- full run ----------------------------------------------------------
    def run_all(
        self, target_date: date, *, top_n: int | None = None, skip_discovery: bool = False
    ) -> RunSummary:
        summary = RunSummary(target_date=target_date)
        if not skip_discovery:
            summary.discovered = len(self.discover(target_date, top_n=top_n))
        else:
            summary.discovered = len(self.videos_for(target_date))
        summary.transcribed, summary.transcript_failures = self.transcribe(target_date)
        summary.analyzed, summary.analysis_failures = self.analyze(target_date)
        summary.report_path, summary.total_cost_usd = self.report(target_date)
        self._auto_prune()
        log.info("run.complete", **summary.model_dump(mode="json"))
        return summary

    def _auto_prune(self) -> None:
        s = self.settings
        if s.transcript_retention_days > 0 or s.report_json_retention_days > 0:
            self.prune()

    # =====================================================================
    # Brief mode: explicit URLs -> transcripts -> analysis -> fact check -> brief
    # =====================================================================
    def ingest_urls(self, urls: list[str], target_date: date) -> list[VideoMeta]:
        ids: list[str] = []
        for u in urls:
            vid = extract_video_id(u)
            if vid is None:
                log.warning("ingest.bad_url", url=u)
                continue
            if vid not in ids:
                ids.append(vid)
        if not ids:
            raise ValueError("no valid YouTube video ids in the given URLs")
        with _StageRun(self.db, target_date, "ingest") as run:
            key = self.settings.youtube_api_key if self.settings.youtube_backend != "ytdlp" else ""
            metas = fetch_metadata_for_ids(ids, key)
            with self.db.session() as s:
                for meta in metas:
                    row = s.get(Video, meta.video_id)
                    if row is None:
                        row = Video(
                            video_id=meta.video_id, target_date=target_date, discovery_source=""
                        )
                        s.add(row)
                    for field in (
                        "title",
                        "channel_id",
                        "channel_title",
                        "published_at",
                        "duration_seconds",
                        "view_count",
                        "like_count",
                        "comment_count",
                        "description",
                    ):
                        setattr(row, field, getattr(meta, field))
                    row.discovery_source = row.discovery_source or "url"
                    row.stats_updated_at = utcnow()
            run.detail.update(requested=len(urls), ingested=len(metas))
        return metas

    def latest_fact_checks(self, video_ids: list[str]) -> dict[str, FactCheck]:
        if not video_ids:
            return {}
        with self.db.session() as s:
            rows = s.scalars(
                select(FactCheck)
                .where(
                    FactCheck.video_id.in_(video_ids),
                    FactCheck.prompt_version == FACTCHECK_PROMPT_VERSION,
                )
                .order_by(FactCheck.created_at)
            ).all()
        latest: dict[str, FactCheck] = {}
        for fc in rows:
            latest[fc.video_id] = fc
        return latest

    def fact_check(
        self, target_date: date, *, video_ids: list[str] | None = None, force: bool = False
    ) -> tuple[int, int]:
        ok = failed = 0
        with _StageRun(self.db, target_date, "factcheck") as run:
            analyses = self.latest_analyses(target_date)
            targets = [
                vid
                for vid, a in analyses.items()
                if a.status == "ok" and a.result and (video_ids is None or vid in video_ids)
            ]
            existing = {} if force else self.latest_fact_checks(targets)
            todo = [
                vid
                for vid in targets
                if not (
                    existing.get(vid)
                    and existing[vid].status == "ok"
                    and existing[vid].analysis_id == analyses[vid].id
                )
            ]
            videos = {v.video_id: _video_meta(v) for v in self.videos_for(target_date)}
            today = target_date.isoformat()

            def _one(vid: str):
                return vid, self.analyzer.fact_check(
                    videos[vid], VideoAnalysis.model_validate(analyses[vid].result), today
                )

            workers = max(1, self.settings.analysis_concurrency)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for fut in as_completed([pool.submit(_one, vid) for vid in todo]):
                    vid, outcome = fut.result()
                    with self.db.session() as s:
                        s.add(
                            FactCheck(
                                video_id=vid,
                                analysis_id=analyses[vid].id,
                                prompt_version=FACTCHECK_PROMPT_VERSION,
                                model=outcome.usage.model,
                                status=outcome.status,
                                result=outcome.result.model_dump() if outcome.result else None,
                                error=outcome.error,
                                input_tokens=outcome.usage.input_tokens,
                                output_tokens=outcome.usage.output_tokens,
                                cache_read_input_tokens=outcome.usage.cache_read_input_tokens,
                                cache_creation_input_tokens=outcome.usage.cache_creation_input_tokens,
                                web_searches=outcome.web_searches,
                                cost_usd=outcome.usage.cost_usd,
                                request_id=outcome.request_id,
                            )
                        )
                    if outcome.status == "ok":
                        ok += 1
                    else:
                        failed += 1
                        log.warning(
                            "factcheck.failed",
                            video_id=vid,
                            status=outcome.status,
                            error=outcome.error[:300],
                        )
            run.detail.update(pending=len(todo), ok=ok, failed=failed)
        return ok, failed

    def brief(
        self,
        target_date: date,
        *,
        video_ids: list[str] | None = None,
        regenerate: bool = True,
    ) -> tuple[str, float]:
        """Assemble analyses + fact checks, generate the trading brief, write files.

        ``regenerate=False`` re-renders from the last stored brief (template changes,
        language switch) without calling the model."""
        with _StageRun(self.db, target_date, "brief") as run:
            videos = [
                v
                for v in self.videos_for(target_date)
                if video_ids is None or v.video_id in video_ids
            ]
            if video_ids:  # preserve the order the user gave
                order = {vid: i for i, vid in enumerate(video_ids)}
                videos.sort(key=lambda v: order.get(v.video_id, 1_000))
            metas = [_video_meta(v) for v in videos]
            ids = [v.video_id for v in videos]
            analysis_rows = self.latest_analyses(target_date)
            fc_rows = self.latest_fact_checks(ids)

            analyses: dict[str, VideoAnalysis] = {}
            fact_checks: dict[str, FactCheckReport] = {}
            failures: dict[str, str] = {}
            cost = 0.0
            with self.db.session() as s:
                for v in videos:
                    a = analysis_rows.get(v.video_id)
                    if a is not None and a.status == "ok" and a.result:
                        analyses[v.video_id] = VideoAnalysis.model_validate(a.result)
                    elif a is not None:
                        failures[v.video_id] = f"analysis {a.status}: {a.error[:160]}"
                    else:
                        t = s.get(Transcript, v.video_id)
                        failures[v.video_id] = (
                            f"transcript {t.status}: {t.error[:160]}" if t else "no transcript yet"
                        )
                    fc = fc_rows.get(v.video_id)
                    if fc is not None and fc.status == "ok" and fc.result:
                        fact_checks[v.video_id] = FactCheckReport.model_validate(fc.result)
                        cost += fc.cost_usd
                    elif fc is not None:
                        failures[f"factcheck:{v.video_id}"] = f"{fc.status}: {fc.error[:160]}"
                    if a is not None:
                        cost += a.cost_usd

            stats = compute_ticker_stats(analyses)
            brief_result: TradingBrief | None = None
            status, error = "skipped", "no successful analyses"
            if analyses and not regenerate:
                with self.db.session() as s:
                    prev = s.scalars(
                        select(Brief)
                        .where(Brief.target_date == target_date, Brief.status == "ok")
                        .order_by(Brief.created_at.desc())
                    ).first()
                if prev is not None and prev.result:
                    brief_result = TradingBrief.model_validate(prev.result)
                    status, error = "ok", ""
                    cost += (
                        prev.cost_usd
                        - sum(a.cost_usd for a in analysis_rows.values())
                        - sum(f.cost_usd for f in fc_rows.values())
                    )
            elif analyses:
                outcome = self.analyzer.trading_brief(
                    target_date.isoformat(), metas, analyses, fact_checks, stats
                )
                status, error = outcome.status, outcome.error
                cost += outcome.usage.cost_usd
                if outcome.status == "ok":
                    brief_result = outcome.result
                else:
                    log.warning("brief.failed", status=outcome.status, error=outcome.error)

            markdown = render_brief_markdown(
                target_date,
                metas,
                analyses,
                fact_checks,
                stats,
                brief_result,
                failures=failures,
                cost_usd=cost,
                language=self.settings.report_language,
            )
            payload = build_brief_json(
                target_date,
                metas,
                analyses,
                fact_checks,
                stats,
                brief_result,
                failures=failures,
                cost_usd=cost,
            )
            path = write_brief(self.settings.reports_dir, target_date, markdown, payload)
            with self.db.session() as s:
                s.add(
                    Brief(
                        target_date=target_date,
                        video_ids=ids,
                        prompt_version=BRIEF_PROMPT_VERSION,
                        model=self.settings.claude_model,
                        status=status,
                        result=brief_result.model_dump() if brief_result else None,
                        markdown=markdown,
                        error=error if status != "ok" else "",
                        cost_usd=cost,
                    )
                )
            run.detail.update(
                videos=len(videos),
                analysed=len(analyses),
                fact_checked=len(fact_checks),
                status=status,
                cost_usd=round(cost, 4),
            )
        return str(path), cost

    def run_brief(
        self,
        urls: list[str],
        target_date: date,
        *,
        fact_check: bool = True,
        include_daily: bool = False,
    ) -> RunSummary:
        summary = RunSummary(target_date=target_date)
        metas = self.ingest_urls(urls, target_date)
        ids = [m.video_id for m in metas]
        if include_daily:
            ids = list(dict.fromkeys(ids + [v.video_id for v in self.videos_for(target_date)]))
        summary.discovered = len(ids)
        summary.transcribed, summary.transcript_failures = self.transcribe(
            target_date, video_ids=ids
        )
        summary.analyzed, summary.analysis_failures = self.analyze(target_date, video_ids=ids)
        if fact_check:
            self.fact_check(target_date, video_ids=ids)
        summary.report_path, summary.total_cost_usd = self.brief(target_date, video_ids=ids)
        self._auto_prune()
        log.info("brief.complete", **summary.model_dump(mode="json"))
        return summary

    # =====================================================================
    # Housekeeping
    # =====================================================================
    def prune(
        self,
        *,
        transcript_days: int | None = None,
        json_days: int | None = None,
        today: date | None = None,
    ) -> dict[str, int]:
        """Drop bulky data we no longer need. Analyses, fact checks and reports are kept.

        - transcript text for videos older than ``transcript_days`` (row stays, status 'pruned')
        - per-day report JSON files older than ``json_days`` (markdown is kept)
        - leftover ``ytstock-*`` temp dirs from interrupted Whisper runs
        """
        import shutil
        import tempfile

        today = today or date.today()
        t_days = (
            self.settings.transcript_retention_days if transcript_days is None else transcript_days
        )
        j_days = self.settings.report_json_retention_days if json_days is None else json_days
        out = {"transcripts_pruned": 0, "json_deleted": 0, "temp_dirs_removed": 0, "bytes_freed": 0}

        if t_days > 0:
            cutoff = today - timedelta(days=t_days)
            with self.db.session() as s:
                ids = s.scalars(
                    select(Video.video_id).where(
                        Video.target_date < cutoff,
                        Video.video_id.in_(
                            select(Transcript.video_id).where(Transcript.status == "ok")
                        ),
                    )
                ).all()
                if ids:
                    freed = s.scalar(
                        select(text("coalesce(sum(length(text)),0)"))
                        .select_from(Transcript)
                        .where(Transcript.video_id.in_(ids))
                    )
                    s.execute(
                        update(Transcript)
                        .where(Transcript.video_id.in_(ids))
                        .values(text="", status="pruned")
                    )
                    out["transcripts_pruned"] = len(ids)
                    out["bytes_freed"] += int(freed or 0)
            if ids and self.db.engine.dialect.name == "sqlite":
                with self.db.engine.connect() as conn:
                    conn.exec_driver_sql("VACUUM")

        if j_days > 0:
            cutoff = today - timedelta(days=j_days)
            for path in self.settings.reports_dir.glob("*/*/*.json"):
                try:
                    day = date.fromisoformat(path.parent.name)
                except ValueError:
                    continue
                if day < cutoff:
                    out["bytes_freed"] += path.stat().st_size
                    path.unlink()
                    out["json_deleted"] += 1

        for tmp in Path(tempfile.gettempdir()).glob("ytstock-*"):
            if tmp.is_dir():
                size = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file())
                shutil.rmtree(tmp, ignore_errors=True)
                out["temp_dirs_removed"] += 1
                out["bytes_freed"] += size

        log.info("prune.done", **out)
        return out
