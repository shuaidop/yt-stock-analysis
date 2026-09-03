from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select

from ytstock.db import Analysis, Brief, DailyReport, FactCheck, PipelineRun, Transcript
from ytstock.discovery import Sources
from ytstock.pipeline import Pipeline
from ytstock.transcripts import TranscriptTransientError, TranscriptUnavailable

from .conftest import FakeClaudeClient, make_video
from .test_discovery import FakeYouTube

TARGET = date(2026, 9, 2)


class FakeTranscripts:
    def __init__(self, behaviour=None):
        self.behaviour = behaviour or {}
        self.calls = []

    def fetch(self, video_id):
        self.calls.append(video_id)
        b = self.behaviour.get(video_id)
        if isinstance(b, Exception):
            raise b
        return SimpleNamespace(
            text=f"transcript for {video_id}",
            language="en",
            source="youtube_captions",
            is_generated=False,
        )


def _pipeline(settings, db, **kw):
    from ytstock.analysis import GroundedAnalyzer

    sources = Sources(
        search_queries=["stock market today"], channels=["@chan"], title_keywords=["market"]
    )
    return Pipeline(
        settings,
        db,
        youtube=kw.get("youtube", FakeYouTube()),
        transcripts=kw.get("transcripts", FakeTranscripts()),
        analyzer=GroundedAnalyzer(settings, client=kw.get("client", FakeClaudeClient())),
        sources=sources,
    )


def test_run_all_end_to_end_and_idempotent(settings, db):
    client = FakeClaudeClient()
    transcripts = FakeTranscripts()
    p = _pipeline(settings, db, client=client, transcripts=transcripts)

    summary = p.run_all(TARGET)
    assert summary.discovered == 2
    assert summary.transcribed == 2 and summary.transcript_failures == 0
    assert summary.analyzed == 2 and summary.analysis_failures == 0
    assert summary.report_path.endswith("2026-09-02.md")
    assert summary.total_cost_usd > 0
    # 2 video analyses + 1 synthesis
    assert len(client.calls) == 3
    md = (settings.reports_dir / "2026-09-02.md").read_text()
    assert "Stock market recap" in md and "Creators lean bullish" in md

    with db.session() as s:
        assert (
            s.scalar(select(DailyReport).where(DailyReport.target_date == TARGET)).video_count == 2
        )
        stages = [r.stage for r in s.scalars(select(PipelineRun).order_by(PipelineRun.id))]
        assert stages == ["discover", "transcribe", "analyze", "report"]
        assert all(r.status == "ok" for r in s.scalars(select(PipelineRun)))

    # second run: no new transcript fetches or video analyses, only the synthesis
    summary2 = p.run_all(TARGET)
    assert summary2.transcribed == 0 and summary2.analyzed == 0
    assert len(transcripts.calls) == 2
    assert len(client.calls) == 4


def test_transcript_failures_are_recorded_and_retried_sensibly(settings, db):
    transcripts = FakeTranscripts(
        {
            "s0000000001": TranscriptUnavailable("TranscriptsDisabled"),
            "s0000000002": TranscriptTransientError("blocked"),
        }
    )
    p = _pipeline(settings, db, transcripts=transcripts)
    p.discover(TARGET)
    ok, failed = p.transcribe(TARGET)
    assert (ok, failed) == (0, 2)
    with db.session() as s:
        rows = {t.video_id: t for t in s.scalars(select(Transcript))}
    assert rows["s0000000001"].status == "unavailable"
    assert rows["s0000000002"].status == "error" and rows["s0000000002"].attempts == 1

    # transient failures get retried; the report explains skipped videos
    transcripts.behaviour["s0000000002"] = None
    ok, failed = p.transcribe(TARGET)
    assert ok == 1
    p.analyze(TARGET)
    path, _ = p.report(TARGET)
    md = Path(path).read_text()
    assert "_Skipped: transcript unavailable: TranscriptsDisabled_" in md


def test_analysis_failure_does_not_abort_day(settings, db):
    from .conftest import fake_response

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return fake_response(None, stop_reason="refusal")
        return FakeClaudeClient._default(request)

    p = _pipeline(settings, db, client=FakeClaudeClient(handler))
    p.settings.analysis_concurrency = 1
    summary = p.run_all(TARGET)
    assert summary.analyzed == 1 and summary.analysis_failures == 1
    with db.session() as s:
        statuses = sorted(a.status for a in s.scalars(select(Analysis)))
    assert statuses == ["ok", "refused"]
    # the refused video is retried on the next analyze pass
    ok, failed = p.analyze(TARGET)
    assert ok == 1 and failed == 0


def test_run_brief_from_urls(settings, db, monkeypatch):
    client = FakeClaudeClient()
    p = _pipeline(settings, db, client=client)
    monkeypatch.setattr(
        "ytstock.pipeline.fetch_metadata_for_ids",
        lambda ids, key: [make_video(v, title=f"Video {v}", discovery_source="url") for v in ids],
    )
    urls = [
        "https://youtu.be/AAAAAAAAAAA",
        "https://www.youtube.com/watch?v=BBBBBBBBBBB",
        "garbage",
    ]
    summary = p.run_brief(urls, TARGET)
    assert summary.discovered == 2 and summary.analyzed == 2
    assert summary.report_path.endswith("brief-2026-09-02.md")
    # 2 analyses + 2 fact checks + 1 brief
    formats = [c["output_format"].__name__ for c in client.calls]
    assert formats.count("VideoAnalysis") == 2
    assert formats.count("FactCheckReport") == 2
    assert formats.count("TradingBrief") == 1
    md = Path(summary.report_path).read_text()
    assert "Long semis into the Fed" in md and "❌ contradicted" in md
    assert md.index("Video AAAAAAAAAAA") < md.index("Video BBBBBBBBBBB")  # user order kept
    with db.session() as s:
        assert s.scalar(select(FactCheck).limit(1)).web_searches == 3
        brief = s.scalar(select(Brief))
        assert brief.status == "ok" and brief.video_ids == ["AAAAAAAAAAA", "BBBBBBBBBBB"]

    # fact checks are not repeated for unchanged analyses
    p.run_brief(urls[:2], TARGET)
    formats = [c["output_format"].__name__ for c in client.calls]
    assert formats.count("FactCheckReport") == 2 and formats.count("TradingBrief") == 2


def test_run_brief_without_factcheck(settings, db, monkeypatch):
    client = FakeClaudeClient()
    p = _pipeline(settings, db, client=client)
    monkeypatch.setattr(
        "ytstock.pipeline.fetch_metadata_for_ids", lambda ids, key: [make_video(v) for v in ids]
    )
    p.run_brief(["CCCCCCCCCCC"], TARGET, fact_check=False)
    assert "FactCheckReport" not in [c["output_format"].__name__ for c in client.calls]
