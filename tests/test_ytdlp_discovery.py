from __future__ import annotations

from datetime import UTC, datetime

from ytstock.ytdlp_discovery import SP_VIEWS_TODAY, YtDlpClient


def _client(**kw):
    return YtDlpClient(
        min_duration=180, max_duration=7200, min_views=1000, exclude_keywords=["nifty"], **kw
    )


def test_search_prefilters_flat_entries(monkeypatch):
    c = _client()
    listings = {}

    def fake_flat(url, n):
        listings[url] = n
        return [
            {
                "id": "good0000001",
                "title": "Stock market today",
                "duration": 600,
                "view_count": 5000,
            },
            {
                "id": "short000001",
                "title": "Stock market today",
                "duration": 30,
                "view_count": 5000,
            },
            {"id": "lowviews001", "title": "Stock market today", "duration": 600, "view_count": 10},
            {"id": "nifty000001", "title": "Nifty live", "duration": 600, "view_count": 99999},
            {
                "id": "live0000001",
                "title": "Market live",
                "duration": None,
                "view_count": None,
                "live_status": "is_live",
            },
            {"id": "unknown0001", "title": "Market recap", "duration": None, "view_count": None},
        ]

    monkeypatch.setattr(c, "_flat_entries", fake_flat)
    ids = c.search_video_ids("stock market today")
    assert ids == ["good0000001", "unknown0001"]
    assert any(SP_VIEWS_TODAY in u for u in listings)
    assert any(u.startswith("ytsearch") for u in listings)


def test_channel_uploads_and_resolve(monkeypatch):
    c = _client()
    seen = {}
    monkeypatch.setattr(
        c,
        "_flat_entries",
        lambda url, n: (
            seen.setdefault("url", url)
            and [
                {"id": "vid00000001", "title": "x", "duration": 600, "view_count": 5},
            ]
        ),
    )
    assert c.resolve_channel_id("@handle") == "@handle"
    assert c.channel_upload_ids("@handle") == ["vid00000001"]  # views not applied for channels
    assert seen["url"] == "https://www.youtube.com/@handle/videos"
    c.channel_upload_ids("UCabc")
    assert seen["url"] == "https://www.youtube.com/@handle/videos"  # setdefault kept first


def test_videos_hydrates_with_cap(monkeypatch):
    c = _client(max_hydrate=2, hydrate_workers=2)
    c._flat_cache = {
        "a0000000001": {"view_count": 10},
        "b0000000001": {"view_count": 300},
        "c0000000001": {"view_count": 200},
    }
    hydrated = []

    def fake_hydrate(vid):
        hydrated.append(vid)
        from ytstock.schemas import VideoMeta

        return VideoMeta(
            video_id=vid,
            title="t",
            channel_id="",
            channel_title="",
            published_at=datetime.now(UTC),
            duration_seconds=1,
        )

    monkeypatch.setattr(c, "_hydrate", fake_hydrate)
    out = c.videos(["a0000000001", "b0000000001", "c0000000001", "b0000000001"])
    assert sorted(hydrated) == ["b0000000001", "c0000000001"]
    assert {m.video_id for m in out} == {"b0000000001", "c0000000001"}


def test_hydrate_maps_fields(monkeypatch):
    c = _client()

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def extract_info(self, url, download=False, process=False):
            return {
                "id": "vid00000001",
                "title": "T",
                "channel_id": "UC1",
                "channel": "Chan",
                "timestamp": 1_788_302_338,
                "duration": 333,
                "view_count": 12,
                "like_count": 3,
                "description": "d",
            }

    import types

    monkeypatch.setitem(
        __import__("sys").modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL)
    )
    m = c._hydrate("vid00000001")
    assert m is not None and m.published_at.year == 2026 and m.duration_seconds == 333
    assert m.published_at.tzinfo is not None
