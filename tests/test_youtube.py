from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from ytstock.youtube import (
    YouTubeAPIError,
    YouTubeClient,
    extract_video_id,
    oembed_metadata,
    parse_iso8601_duration,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("PT1H2M3S", 3723), ("PT15M", 900), ("PT45S", 45), ("P1DT1H", 90000), ("", 0), ("junk", 0)],
)
def test_parse_duration(value, expected):
    assert parse_iso8601_duration(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=10", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ?feature=share", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com/nope", None),
        ("", None),
    ],
)
def test_extract_video_id(value, expected):
    assert extract_video_id(value) == expected


def _transport(routes):
    def handler(request: httpx.Request) -> httpx.Response:
        for path, fn in routes.items():
            if request.url.path.endswith(path):
                return fn(request)
        return httpx.Response(404, json={"error": {"message": "nope"}})

    return httpx.MockTransport(handler)


def test_search_and_videos_roundtrip():
    seen = []

    def search(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"items": [{"id": {"videoId": "vid00000001"}}]})

    def videos(request):
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "vid00000001",
                        "snippet": {
                            "title": "Market recap",
                            "channelId": "UC1",
                            "channelTitle": "Chan",
                            "publishedAt": "2026-09-02T21:00:00Z",
                            "description": "d",
                        },
                        "contentDetails": {"duration": "PT12M"},
                        "statistics": {"viewCount": "1234", "likeCount": "5"},
                    }
                ]
            },
        )

    client = YouTubeClient("k", transport=_transport({"/search": search, "/videos": videos}))
    ids = client.search_video_ids(
        "stock market",
        published_after=datetime(2026, 9, 2, tzinfo=UTC),
        published_before=datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert ids == ["vid00000001"]
    assert {s["videoDuration"] for s in seen} == {"medium", "long"}
    assert seen[0]["publishedAfter"].endswith("Z")
    assert client.quota_used == 200

    metas = client.videos(ids)
    assert metas[0].duration_seconds == 720
    assert metas[0].view_count == 1234
    assert metas[0].published_at.tzinfo is not None
    assert client.quota_used == 201


def test_quota_error_is_flagged():
    def search(_):
        return httpx.Response(
            403,
            json={"error": {"message": "quota", "errors": [{"reason": "quotaExceeded"}]}},
        )

    client = YouTubeClient("k", transport=_transport({"/search": search}))
    with pytest.raises(YouTubeAPIError) as exc:
        client.search_video_ids(
            "x",
            published_after=datetime(2026, 9, 2, tzinfo=UTC),
            published_before=datetime(2026, 9, 3, tzinfo=UTC),
        )
    assert exc.value.is_quota


def test_channel_uploads_stops_at_window():
    def playlist(request):
        assert request.url.params["playlistId"] == "UUabc"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "contentDetails": {
                            "videoId": "new00000001",
                            "videoPublishedAt": "2026-09-02T22:00:00Z",
                        }
                    },
                    {
                        "contentDetails": {
                            "videoId": "old00000001",
                            "videoPublishedAt": "2026-08-01T22:00:00Z",
                        }
                    },
                ],
                "nextPageToken": "should-not-follow",
            },
        )

    client = YouTubeClient("k", transport=_transport({"/playlistItems": playlist}))
    ids = client.channel_upload_ids(
        "UCabc",
        published_after=datetime(2026, 9, 2, tzinfo=UTC),
        published_before=datetime(2026, 9, 3, tzinfo=UTC),
    )
    assert ids == ["new00000001"]
    assert client.quota_used == 1


def test_missing_api_key_rejected():
    with pytest.raises(ValueError):
        YouTubeClient("")


def test_oembed_metadata():
    def handler(request):
        assert "oembed" in request.url.path
        return httpx.Response(200, json={"title": "T", "author_name": "A"})

    meta = oembed_metadata("dQw4w9WgXcQ", transport=httpx.MockTransport(handler))
    assert meta.title == "T" and meta.channel_title == "A" and meta.discovery_source == "url"
    assert json.dumps(meta.model_dump(mode="json"))
