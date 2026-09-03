"""Keyless discovery via yt-dlp (no Google Cloud project needed).

Implements the same four methods ``discovery.discover`` calls on
``YouTubeClient`` so the two backends are interchangeable:
``search_video_ids``, ``resolve_channel_id``, ``channel_upload_ids``, ``videos``.

Flat search/channel listings are cheap (<1 s) but lack publish dates, so
candidates are pre-filtered on the flat fields and then hydrated per video
(~2-3 s each, run in a thread pool) to get timestamps and full stats.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from ytstock.log import get_logger
from ytstock.schemas import VideoMeta

log = get_logger(__name__)

# YouTube search-filter tokens (the `sp` query parameter).
SP_VIEWS_TODAY = "CAMSAggC"  # sort: view count, upload date: today
SP_VIEWS_WEEK = "CAMSAggD"  # sort: view count, upload date: this week


class YtDlpClient:
    def __init__(
        self,
        *,
        region_code: str = "US",
        proxy_url: str = "",
        relevance_results: int = 15,
        filtered_results: int = 20,
        min_duration: int = 0,
        max_duration: int = 10**9,
        min_views: int = 0,
        exclude_keywords: Sequence[str] = (),
        hydrate_workers: int = 6,
        max_hydrate: int = 150,
    ) -> None:
        self._region = region_code
        self._proxy = proxy_url
        self._relevance_results = relevance_results
        self._filtered_results = filtered_results
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._min_views = min_views
        self._exclude = [k.lower() for k in exclude_keywords]
        self._workers = hydrate_workers
        self._max_hydrate = max_hydrate
        self.quota_used = 0  # request count, for log parity with the API backend
        self._flat_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    def _ydl(self, *, flat: bool, playlist_items: str | None = None):
        import yt_dlp  # local import keeps module import cheap

        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noprogress": True,
            "extract_flat": flat,
            "socket_timeout": 20,
            "geo_bypass_country": self._region,
            "extractor_args": {"youtube": {"lang": ["en"]}},
        }
        if playlist_items:
            opts["playlist_items"] = playlist_items
        if self._proxy:
            opts["proxy"] = self._proxy
        return yt_dlp.YoutubeDL(opts)

    def _flat_entries(self, url: str, n: int) -> list[dict[str, Any]]:
        self.quota_used += 1
        try:
            with self._ydl(flat=True, playlist_items=f"1-{n}") as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            log.warning("ytdlp.listing_failed", url=url, error=str(exc)[:200])
            return []
        entries = [e for e in (info or {}).get("entries", []) if e and e.get("id")]
        for e in entries:
            self._flat_cache.setdefault(e["id"], e)
        return entries

    def _passes_flat_filter(self, e: dict[str, Any], *, apply_views: bool) -> bool:
        title = (e.get("title") or "").lower()
        if any(k in title for k in self._exclude):
            return False
        dur = e.get("duration")
        if dur is not None and not (self._min_duration <= dur <= self._max_duration):
            return False
        views = e.get("view_count")
        if apply_views and views is not None and views < self._min_views:
            return False
        return e.get("live_status") not in {"is_live", "is_upcoming"}

    # ------------------------------------------------------------------ #
    def search_video_ids(self, query: str, **_: Any) -> list[str]:
        q = query.replace(" ", "+")
        listings = [
            (f"ytsearch{self._relevance_results}:{query}", self._relevance_results),
            (
                f"https://www.youtube.com/results?search_query={q}&sp={SP_VIEWS_TODAY}",
                self._filtered_results,
            ),
        ]
        ids: list[str] = []
        for url, n in listings:
            for e in self._flat_entries(url, n):
                if self._passes_flat_filter(e, apply_views=True):
                    ids.append(e["id"])
        log.debug("ytdlp.search", query=query, candidates=len(ids))
        return list(dict.fromkeys(ids))

    def resolve_channel_id(self, handle_or_id: str) -> str:
        return handle_or_id  # both '@handle' and 'UC…' work in channel URLs

    def channel_upload_ids(self, channel: str, *, max_items: int = 15, **_: Any) -> list[str]:
        path = channel if channel.startswith("@") else f"channel/{channel}"
        entries = self._flat_entries(f"https://www.youtube.com/{path}/videos", max_items)
        # Uploads are newest-first; date filtering happens after hydration.
        return [e["id"] for e in entries if self._passes_flat_filter(e, apply_views=False)]

    # ------------------------------------------------------------------ #
    def _hydrate(self, video_id: str) -> VideoMeta | None:
        self.quota_used += 1
        try:
            with self._ydl(flat=False) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False, process=False
                )
        except Exception as exc:
            log.debug("ytdlp.hydrate_failed", video_id=video_id, error=str(exc)[:200])
            return None
        if not info:
            return None
        ts = info.get("timestamp") or info.get("release_timestamp")
        if ts is None:
            return None
        return VideoMeta(
            video_id=video_id,
            title=info.get("title") or video_id,
            channel_id=info.get("channel_id") or "",
            channel_title=info.get("channel") or info.get("uploader") or "",
            published_at=datetime.fromtimestamp(int(ts), tz=UTC),
            duration_seconds=int(info.get("duration") or 0),
            view_count=int(info.get("view_count") or 0),
            like_count=int(info.get("like_count") or 0),
            comment_count=int(info.get("comment_count") or 0),
            description=info.get("description") or "",
        )

    def videos(self, video_ids: Sequence[str]) -> list[VideoMeta]:
        ids = list(dict.fromkeys(video_ids))
        if len(ids) > self._max_hydrate:
            # keep the most-viewed by flat data when we must cut
            ids.sort(key=lambda v: -(self._flat_cache.get(v, {}).get("view_count") or 0))
            ids = ids[: self._max_hydrate]
        log.info("ytdlp.hydrate", count=len(ids), workers=self._workers)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            results = list(pool.map(self._hydrate, ids))
        return [m for m in results if m is not None]

    def close(self) -> None:  # parity with YouTubeClient
        return None
