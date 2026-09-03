"""Transcript acquisition.

Primary: YouTube captions via ``youtube-transcript-api`` (manual first, then
auto-generated, then any translatable track -> English).
Fallback (opt-in): download audio with yt-dlp and transcribe with faster-whisper.

Cloud provider IPs are frequently blocked by YouTube for caption requests, so a
proxy (generic or Webshare) can be configured; see ``Settings``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
)
from youtube_transcript_api.proxies import GenericProxyConfig, ProxyConfig, WebshareProxyConfig

from ytstock.config import Settings
from ytstock.log import get_logger
from ytstock.schemas import TranscriptResult

log = get_logger(__name__)


class TranscriptUnavailable(Exception):
    """Permanent: no captions exist and no fallback produced text."""


class TranscriptTransientError(Exception):
    """Retryable: rate-limit / IP block / network problem."""


class TranscriptProvider(Protocol):
    def fetch(self, video_id: str) -> TranscriptResult: ...


def build_proxy_config(settings: Settings) -> ProxyConfig | None:
    if settings.webshare_proxy_username and settings.webshare_proxy_password:
        return WebshareProxyConfig(
            proxy_username=settings.webshare_proxy_username,
            proxy_password=settings.webshare_proxy_password,
        )
    if settings.yt_proxy_url:
        return GenericProxyConfig(http_url=settings.yt_proxy_url, https_url=settings.yt_proxy_url)
    return None


def _join_snippets(fetched) -> str:
    parts = [snippet.text.strip() for snippet in fetched]
    return " ".join(p for p in parts if p)


class CaptionProvider:
    """Fetches existing caption tracks. Never downloads media."""

    def __init__(self, languages: list[str], proxy_config: ProxyConfig | None = None, api=None):
        self._languages = languages
        self._api = api or YouTubeTranscriptApi(proxy_config=proxy_config)

    @retry(
        retry=retry_if_exception_type(TranscriptTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def fetch(self, video_id: str) -> TranscriptResult:
        try:
            transcript_list = self._api.list(video_id)
        except (RequestBlocked, IpBlocked, YouTubeRequestFailed) as exc:
            raise TranscriptTransientError(str(exc)) from exc
        except (TranscriptsDisabled, VideoUnavailable, CouldNotRetrieveTranscript) as exc:
            raise TranscriptUnavailable(exc.__class__.__name__) from exc

        # 1) manual English, 2) generated English
        for finder, generated in (
            (transcript_list.find_manually_created_transcript, False),
            (transcript_list.find_generated_transcript, True),
        ):
            try:
                track = finder(self._languages)
            except NoTranscriptFound:
                continue
            return TranscriptResult(
                video_id=video_id,
                text=_join_snippets(self._safe_fetch(track)),
                language=track.language_code,
                source="youtube_captions",
                is_generated=generated,
            )

        # 3) any other language, untranslated: the analysis model reads it natively and
        #    YouTube's machine translation would only lose information.
        tracks = sorted(transcript_list, key=lambda t: t.is_generated)  # manual first
        for track in tracks:
            return TranscriptResult(
                video_id=video_id,
                text=_join_snippets(self._safe_fetch(track)),
                language=track.language_code,
                source="youtube_captions",
                is_generated=track.is_generated,
            )
        raise TranscriptUnavailable("NoTranscriptFound")

    @staticmethod
    def _safe_fetch(track):
        try:
            return track.fetch()
        except (RequestBlocked, IpBlocked, YouTubeRequestFailed) as exc:
            raise TranscriptTransientError(str(exc)) from exc


class WhisperProvider:
    """yt-dlp + faster-whisper. Heavy; only used when explicitly enabled."""

    def __init__(self, model_name: str = "small", proxy_url: str = "", beam_size: int = 1) -> None:
        self._model_name = model_name
        self._proxy_url = proxy_url
        self._beam_size = beam_size
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy: optional extra

            self._model = WhisperModel(self._model_name, compute_type="int8", cpu_threads=8)
        return self._model

    def fetch(self, video_id: str) -> TranscriptResult:
        import yt_dlp  # lazy: optional extra

        with tempfile.TemporaryDirectory(prefix="ytstock-") as tmp:
            outtmpl = str(Path(tmp) / "%(id)s.%(ext)s")
            opts = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "noprogress": True,
                "postprocessors": [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "m4a"},
                ],
            }
            if self._proxy_url:
                opts["proxy"] = self._proxy_url
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            audio = next(Path(tmp).glob(f"{video_id}.*"), None)
            if audio is None:
                raise TranscriptUnavailable("yt-dlp produced no audio file")
            log.info("whisper.start", video_id=video_id, model=self._model_name)
            segments, info = self._load().transcribe(
                str(audio), vad_filter=True, beam_size=self._beam_size
            )
            text = " ".join(seg.text.strip() for seg in segments)
            log.info(
                "whisper.done",
                video_id=video_id,
                language=getattr(info, "language", None),
                chars=len(text),
            )
        if not text.strip():
            raise TranscriptUnavailable("whisper produced empty transcript")
        return TranscriptResult(
            video_id=video_id,
            text=text,
            language=getattr(info, "language", "en") or "en",
            source="whisper",
            is_generated=True,
        )


class TranscriptService:
    """Runs providers in order; first success wins."""

    def __init__(self, providers: list[TranscriptProvider]) -> None:
        if not providers:
            raise ValueError("at least one transcript provider is required")
        self._providers = providers

    @classmethod
    def from_settings(cls, settings: Settings) -> TranscriptService:
        providers: list[TranscriptProvider] = [
            CaptionProvider(settings.transcript_languages, build_proxy_config(settings))
        ]
        if settings.whisper_fallback:
            providers.append(
                WhisperProvider(
                    settings.whisper_model, settings.yt_proxy_url, settings.whisper_beam_size
                )
            )
        return cls(providers)

    def fetch(self, video_id: str) -> TranscriptResult:
        last_error: Exception | None = None
        for provider in self._providers:
            name = provider.__class__.__name__
            try:
                result = provider.fetch(video_id)
            except TranscriptUnavailable as exc:
                log.info(
                    "transcript.unavailable", video_id=video_id, provider=name, reason=str(exc)
                )
                last_error = exc
                continue
            except TranscriptTransientError as exc:
                log.warning(
                    "transcript.transient", video_id=video_id, provider=name, error=str(exc)
                )
                last_error = exc
                continue
            if result.text.strip():
                return result
            last_error = TranscriptUnavailable(f"{name} returned empty text")
        assert last_error is not None
        raise last_error
