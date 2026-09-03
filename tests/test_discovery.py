from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ytstock.discovery import (
    DiscoveryWindow,
    Sources,
    default_target_date,
    discover,
    filter_and_rank,
)

from .conftest import make_video

NY = ZoneInfo("America/New_York")


def test_window_is_market_day_clamped_to_now():
    now = datetime(2026, 9, 2, 15, 0, tzinfo=NY)
    w = DiscoveryWindow.for_date(date(2026, 9, 2), "America/New_York", now=now)
    assert w.start == datetime(2026, 9, 2, 0, 0, tzinfo=NY)
    assert w.end == now


def test_default_target_date_rolls_back_overnight():
    assert default_target_date(
        "America/New_York", now=datetime(2026, 9, 3, 1, 0, tzinfo=NY)
    ) == date(2026, 9, 2)
    assert default_target_date(
        "America/New_York", now=datetime(2026, 9, 3, 9, 0, tzinfo=NY)
    ) == date(2026, 9, 3)


def test_sources_load():
    s = Sources.load(Path(__file__).parent.parent / "configs" / "sources.yaml")
    assert s.search_queries and s.channels and "crypto" in s.exclude_keywords


def test_filter_and_rank(settings):
    sources = Sources(title_keywords=["market"], exclude_keywords=["crypto"])
    window = DiscoveryWindow.for_date(
        date(2026, 9, 2), "America/New_York", now=datetime(2026, 9, 3, 12, tzinfo=NY)
    )
    vids = [
        make_video("a0000000001", view_count=10, title="market"),  # below min views
        make_video(
            "b0000000001", view_count=5000, title="Market close", duration_seconds=60
        ),  # short
        make_video("c0000000001", view_count=5000, title="Crypto market"),  # excluded
        make_video("d0000000001", view_count=5000, title="Cats"),  # no keyword
        make_video(
            "e0000000001", view_count=5000, title="Cats", discovery_source="channel:@x"
        ),  # channel bypass
        make_video("f0000000001", view_count=9000, title="Market recap"),
        make_video("f0000000001", view_count=9000, title="Market recap"),  # duplicate
        make_video(
            "g0000000001",
            view_count=9999,
            title="Market recap",
            published_at=datetime(2026, 9, 1, 12, tzinfo=UTC),
        ),
    ]
    out = filter_and_rank(vids, settings=settings, sources=sources, window=window, top_n=5)
    assert [v.video_id for v in out] == ["f0000000001", "e0000000001"]


class FakeYouTube:
    quota_used = 0

    def __init__(self):
        self.videos_requested = None

    def search_video_ids(self, query, **_):
        return ["s0000000001", "s0000000002"] if "today" in query else []

    def resolve_channel_id(self, h):
        return "UCresolved"

    def channel_upload_ids(self, channel_id, **_):
        return ["s0000000002", "c0000000001"]

    def videos(self, ids):
        self.videos_requested = list(ids)
        return [
            make_video("s0000000001", view_count=100_000, title="Stock market today"),
            make_video("s0000000002", view_count=200_000, title="Stock market recap"),
            make_video("c0000000001", view_count=50, title="Random vlog"),
        ]


def test_discover_merges_sources_and_marks_channel_precedence(settings):
    sources = Sources(
        search_queries=["stock market today", "nothing"],
        channels=["@chan"],
        title_keywords=["market"],
    )
    window = DiscoveryWindow.for_date(
        date(2026, 9, 2), "America/New_York", now=datetime(2026, 9, 3, 12, tzinfo=NY)
    )
    yt = FakeYouTube()
    out = discover(yt, settings=settings, sources=sources, window=window, top_n=10)
    assert sorted(yt.videos_requested) == ["c0000000001", "s0000000001", "s0000000002"]
    assert [v.video_id for v in out] == ["s0000000002", "s0000000001"]
    assert out[0].discovery_source == "channel:@chan"
    assert out[1].discovery_source == "search:stock market today"
