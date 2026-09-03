"""Find the day's most popular stock-market videos.

Candidates come from (a) keyword searches and (b) a channel watchlist, are
hydrated with statistics, filtered (duration, views, keywords), de-duplicated,
and ranked by view count. Ranking is deterministic so a re-run for the same
day produces the same ordering given the same statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from ytstock.config import Settings
from ytstock.log import get_logger
from ytstock.schemas import VideoMeta
from ytstock.youtube import YouTubeAPIError, YouTubeClient

log = get_logger(__name__)


@dataclass
class Sources:
    search_queries: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    title_keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Sources:
        data = yaml.safe_load(path.read_text()) or {}
        return cls(
            search_queries=[str(q) for q in data.get("search_queries", [])],
            channels=[str(c) for c in data.get("channels", [])],
            title_keywords=[str(k).lower() for k in data.get("title_keywords", [])],
            exclude_keywords=[str(k).lower() for k in data.get("exclude_keywords", [])],
        )


@dataclass(frozen=True)
class DiscoveryWindow:
    target_date: date
    start: datetime
    end: datetime

    @classmethod
    def for_date(
        cls, target_date: date, tz_name: str, now: datetime | None = None
    ) -> DiscoveryWindow:
        tz = ZoneInfo(tz_name)
        start = datetime.combine(target_date, time.min, tzinfo=tz)
        end = start + timedelta(days=1)
        now = now or datetime.now(tz)
        return cls(target_date=target_date, start=start, end=min(end, now.astimezone(tz)))


def default_target_date(tz_name: str, now: datetime | None = None) -> date:
    """Today in market time. Before 06:00 local we assume the caller wants yesterday's
    recap videos (the usual case for an overnight cron)."""
    tz = ZoneInfo(tz_name)
    now = (now or datetime.now(tz)).astimezone(tz)
    if now.hour < 6:
        return (now - timedelta(days=1)).date()
    return now.date()


def _title_ok(title: str, sources: Sources, *, require_keywords: bool) -> bool:
    lowered = title.lower()
    if any(k in lowered for k in sources.exclude_keywords):
        return False
    if require_keywords and sources.title_keywords:
        return any(k in lowered for k in sources.title_keywords)
    return True


def filter_and_rank(
    candidates: list[VideoMeta],
    *,
    settings: Settings,
    sources: Sources,
    window: DiscoveryWindow,
    top_n: int,
) -> list[VideoMeta]:
    seen: set[str] = set()
    kept: list[VideoMeta] = []
    for v in candidates:
        if v.video_id in seen:
            continue
        seen.add(v.video_id)
        if not (window.start <= v.published_at < window.end):
            continue
        if not (
            settings.min_duration_seconds <= v.duration_seconds <= settings.max_duration_seconds
        ):
            continue
        if v.view_count < settings.min_view_count:
            continue
        from_channel = v.discovery_source.startswith("channel:")
        if not _title_ok(v.title, sources, require_keywords=not from_channel):
            continue
        kept.append(v)
    kept.sort(key=lambda v: (-v.view_count, v.published_at, v.video_id))
    return kept[:top_n]


def discover(
    client: YouTubeClient,
    *,
    settings: Settings,
    sources: Sources,
    window: DiscoveryWindow,
    top_n: int | None = None,
) -> list[VideoMeta]:
    """Return the top-N videos for the window. Raises on quota exhaustion."""
    id_sources: dict[str, str] = {}

    for query in sources.search_queries:
        try:
            ids = client.search_video_ids(
                query,
                published_after=window.start,
                published_before=window.end,
                region_code=settings.region_code,
                relevance_language=settings.relevance_language,
            )
        except YouTubeAPIError as exc:
            if exc.is_quota:
                raise
            log.warning("discover.search_failed", query=query, error=str(exc))
            continue
        for vid in ids:
            id_sources.setdefault(vid, f"search:{query}")

    for channel in sources.channels:
        try:
            channel_id = client.resolve_channel_id(channel)
            ids = client.channel_upload_ids(
                channel_id, published_after=window.start, published_before=window.end
            )
        except YouTubeAPIError as exc:
            if exc.is_quota:
                raise
            log.warning("discover.channel_failed", channel=channel, error=str(exc))
            continue
        for vid in ids:
            id_sources[vid] = f"channel:{channel}"  # channel wins: bypasses keyword filter

    log.info("discover.candidates", count=len(id_sources), quota_used=client.quota_used)
    if not id_sources:
        return []

    hydrated = client.videos(list(id_sources))
    for v in hydrated:
        v.discovery_source = id_sources.get(v.video_id, "")

    ranked = filter_and_rank(
        hydrated,
        settings=settings,
        sources=sources,
        window=window,
        top_n=top_n or settings.top_n_videos,
    )
    log.info(
        "discover.ranked",
        kept=len(ranked),
        hydrated=len(hydrated),
        quota_used=client.quota_used,
    )
    return ranked
