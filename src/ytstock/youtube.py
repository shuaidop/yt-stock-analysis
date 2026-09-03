"""Minimal YouTube Data API v3 client (search, playlist items, video details).

Uses plain ``httpx`` rather than the Google client library: three endpoints,
explicit quota accounting, and trivial to mock in tests.

Quota reference (default 10,000 units/day):
  search.list = 100, playlistItems.list = 1, videos.list = 1, channels.list = 1.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ytstock.log import get_logger
from ytstock.schemas import VideoMeta

log = get_logger(__name__)

BASE_URL = "https://www.googleapis.com/youtube/v3"
_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)?$"
)


class YouTubeAPIError(RuntimeError):
    def __init__(self, status: int, message: str, reason: str = "") -> None:
        super().__init__(f"YouTube API {status}: {message}")
        self.status = status
        self.reason = reason

    @property
    def is_quota(self) -> bool:
        return self.reason in {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}


def parse_iso8601_duration(value: str) -> int:
    """'PT1H2M3S' -> 3723. Returns 0 for empty/unparseable values."""
    m = _DURATION_RE.match(value or "")
    if not m:
        return 0
    parts = {k: int(v) for k, v in m.groupdict().items() if v}
    return (
        parts.get("days", 0) * 86400
        + parts.get("h", 0) * 3600
        + parts.get("m", 0) * 60
        + parts.get("s", 0)
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | httpx.TimeoutException):
        return True
    return isinstance(exc, YouTubeAPIError) and exc.status >= 500


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class YouTubeClient:
    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY is required for discovery.")
        self._key = api_key
        self._http = httpx.Client(base_url=BASE_URL, timeout=timeout, transport=transport)
        self.quota_used = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> YouTubeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _get(self, path: str, params: dict, cost: int) -> dict:
        resp = self._http.get(path, params={**params, "key": self._key})
        self.quota_used += cost
        if resp.status_code >= 400:
            reason = ""
            message = resp.text[:500]
            try:
                err = resp.json()["error"]
                message = err.get("message", message)
                reason = (err.get("errors") or [{}])[0].get("reason", "")
            except Exception:
                pass
            raise YouTubeAPIError(resp.status_code, message, reason)
        return resp.json()

    # ------------------------------------------------------------------ #
    def search_video_ids(
        self,
        query: str,
        *,
        published_after: datetime,
        published_before: datetime,
        max_results: int = 50,
        order: str = "viewCount",
        region_code: str = "US",
        relevance_language: str = "en",
    ) -> list[str]:
        params = {
            "part": "id",
            "type": "video",
            "q": query,
            "order": order,
            "maxResults": min(max_results, 50),
            "publishedAfter": _rfc3339(published_after),
            "publishedBefore": _rfc3339(published_before),
            "regionCode": region_code,
            "relevanceLanguage": relevance_language,
            "videoDuration": "medium",  # 4-20 min; 'long' fetched separately below
        }
        ids: list[str] = []
        for duration in ("medium", "long"):
            data = self._get("/search", {**params, "videoDuration": duration}, cost=100)
            ids.extend(item["id"]["videoId"] for item in data.get("items", []))
        log.debug("youtube.search", query=query, results=len(ids))
        return list(dict.fromkeys(ids))

    def resolve_channel_id(self, handle_or_id: str) -> str:
        """Accepts 'UC...' ids or '@handle'. Costs 1 unit for handles."""
        if handle_or_id.startswith("UC") and len(handle_or_id) == 24:
            return handle_or_id
        handle = handle_or_id.lstrip("@")
        data = self._get("/channels", {"part": "id", "forHandle": handle}, cost=1)
        items = data.get("items", [])
        if not items:
            raise YouTubeAPIError(404, f"channel handle not found: {handle_or_id}", "notFound")
        return items[0]["id"]

    def channel_upload_ids(
        self,
        channel_id: str,
        *,
        published_after: datetime,
        published_before: datetime,
        max_pages: int = 2,
    ) -> list[str]:
        """Recent uploads via the channel's uploads playlist (1 unit/page vs 100 for search)."""
        playlist_id = "UU" + channel_id[2:]
        ids: list[str] = []
        page_token: str | None = None
        for _ in range(max_pages):
            params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
            if page_token:
                params["pageToken"] = page_token
            try:
                data = self._get("/playlistItems", params, cost=1)
            except YouTubeAPIError as exc:
                if exc.status == 404:
                    log.warning("youtube.channel_uploads.missing", channel_id=channel_id)
                    return []
                raise
            stop = False
            for item in data.get("items", []):
                published = datetime.fromisoformat(
                    item["contentDetails"]["videoPublishedAt"].replace("Z", "+00:00")
                )
                if published < published_after:
                    stop = True
                    break
                if published <= published_before:
                    ids.append(item["contentDetails"]["videoId"])
            page_token = data.get("nextPageToken")
            if stop or not page_token:
                break
        return ids

    def videos(self, video_ids: Sequence[str]) -> list[VideoMeta]:
        """Hydrate ids with snippet + statistics + duration (1 unit per 50 ids)."""
        out: list[VideoMeta] = []
        for batch in _chunks(video_ids, 50):
            data = self._get(
                "/videos",
                {
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(batch),
                    "maxResults": 50,
                },
                cost=1,
            )
            for item in data.get("items", []):
                sn, st, cd = item["snippet"], item.get("statistics", {}), item["contentDetails"]
                out.append(
                    VideoMeta(
                        video_id=item["id"],
                        title=sn.get("title", ""),
                        channel_id=sn.get("channelId", ""),
                        channel_title=sn.get("channelTitle", ""),
                        published_at=datetime.fromisoformat(
                            sn["publishedAt"].replace("Z", "+00:00")
                        ),
                        duration_seconds=parse_iso8601_duration(cd.get("duration", "")),
                        view_count=int(st.get("viewCount", 0) or 0),
                        like_count=int(st.get("likeCount", 0) or 0),
                        comment_count=int(st.get("commentCount", 0) or 0),
                        description=sn.get("description", "") or "",
                    )
                )
        return out


def _chunks(seq: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# --------------------------------------------------------------------------- #
# Explicit URLs (brief mode)
# --------------------------------------------------------------------------- #
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_PATTERNS = (
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"/(?:shorts|live|embed|v)/([A-Za-z0-9_-]{11})"),
)


def extract_video_id(url_or_id: str) -> str | None:
    s = url_or_id.strip()
    if _VIDEO_ID_RE.match(s):
        return s
    for pat in _URL_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    return None


def oembed_metadata(video_id: str, *, transport: httpx.BaseTransport | None = None) -> VideoMeta:
    """Keyless title/channel lookup. No stats or duration."""
    with httpx.Client(timeout=20.0, transport=transport) as http:
        resp = http.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
        )
    if resp.status_code >= 400:
        raise YouTubeAPIError(resp.status_code, f"oembed failed for {video_id}")
    data = resp.json()
    return VideoMeta(
        video_id=video_id,
        title=data.get("title", video_id),
        channel_id="",
        channel_title=data.get("author_name", ""),
        published_at=datetime.now(tz=UTC),
        duration_seconds=0,
        discovery_source="url",
    )


def ytdlp_metadata(video_id: str) -> VideoMeta:
    """Full metadata without an API key, if the optional yt-dlp extra is installed."""
    import yt_dlp  # lazy: optional extra

    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "noprogress": True}) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    upload = info.get("timestamp")
    published = datetime.fromtimestamp(upload, tz=UTC) if upload else datetime.now(tz=UTC)
    return VideoMeta(
        video_id=video_id,
        title=info.get("title") or video_id,
        channel_id=info.get("channel_id") or "",
        channel_title=info.get("channel") or info.get("uploader") or "",
        published_at=published,
        duration_seconds=int(info.get("duration") or 0),
        view_count=int(info.get("view_count") or 0),
        like_count=int(info.get("like_count") or 0),
        comment_count=int(info.get("comment_count") or 0),
        description=info.get("description") or "",
        discovery_source="url",
    )


def fetch_metadata_for_ids(video_ids: Sequence[str], api_key: str = "") -> list[VideoMeta]:
    """Best available metadata: Data API -> yt-dlp -> oEmbed."""
    if api_key:
        with YouTubeClient(api_key) as client:
            metas = client.videos(video_ids)
        for m in metas:
            m.discovery_source = "url"
        found = {m.video_id for m in metas}
        missing = [v for v in video_ids if v not in found]
    else:
        metas, missing = [], list(video_ids)
    for vid in missing:
        try:
            metas.append(ytdlp_metadata(vid))
            continue
        except ImportError:
            pass
        except Exception as exc:
            log.warning("metadata.ytdlp_failed", video_id=vid, error=str(exc))
        metas.append(oembed_metadata(vid))
    order = {v: i for i, v in enumerate(video_ids)}
    metas.sort(key=lambda m: order.get(m.video_id, 1_000))
    return metas
