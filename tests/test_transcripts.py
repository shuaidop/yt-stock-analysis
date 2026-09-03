from __future__ import annotations

from types import SimpleNamespace

import pytest
from youtube_transcript_api._errors import NoTranscriptFound, RequestBlocked, TranscriptsDisabled

from ytstock.transcripts import (
    CaptionProvider,
    TranscriptService,
    TranscriptTransientError,
    TranscriptUnavailable,
)


def _snips(*texts):
    return [SimpleNamespace(text=t) for t in texts]


class _Track:
    def __init__(self, code="en", generated=False, translatable=True, texts=("hello", "world")):
        self.language_code = code
        self.is_generated = generated
        self.is_translatable = translatable
        self._texts = texts

    def fetch(self):
        return _snips(*self._texts)

    def translate(self, lang):
        return _Track(code=lang, generated=self.is_generated, texts=("hola->hello",))


class _TranscriptList:
    def __init__(self, manual=None, generated=None, others=()):
        self._manual, self._generated, self._others = manual, generated, list(others)

    def find_manually_created_transcript(self, langs):
        if self._manual:
            return self._manual
        raise NoTranscriptFound("v", langs, {})

    def find_generated_transcript(self, langs):
        if self._generated:
            return self._generated
        raise NoTranscriptFound("v", langs, {})

    def __iter__(self):
        return iter(self._others)


class _Api:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    def list(self, video_id):
        if self._exc:
            raise self._exc
        return self._result


def test_prefers_manual_captions():
    p = CaptionProvider(
        ["en"], api=_Api(_TranscriptList(manual=_Track(), generated=_Track(generated=True)))
    )
    r = p.fetch("v")
    assert r.text == "hello world" and r.source == "youtube_captions" and not r.is_generated


def test_falls_back_to_generated_then_native_language():
    p = CaptionProvider(["en"], api=_Api(_TranscriptList(generated=_Track(generated=True))))
    assert p.fetch("v").is_generated

    p = CaptionProvider(
        ["en"],
        api=_Api(
            _TranscriptList(others=[_Track(code="es", generated=True), _Track(code="zh-Hans")])
        ),
    )
    r = p.fetch("v")
    assert r.source == "youtube_captions" and r.language == "zh-Hans"  # manual beats generated
    assert r.text == "hello world" and not r.is_generated


def test_unavailable_and_transient_mapping():
    p = CaptionProvider(["en"], api=_Api(exc=TranscriptsDisabled("v")))
    with pytest.raises(TranscriptUnavailable):
        p.fetch("v")

    p = CaptionProvider(["en"], api=_Api(exc=RequestBlocked("v")))
    p.fetch.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    with pytest.raises(TranscriptTransientError):
        p.fetch("v")


def test_service_falls_through_providers():
    class Bad:
        def fetch(self, vid):
            raise TranscriptUnavailable("none")

    class Good:
        def fetch(self, vid):
            return SimpleNamespace(
                text="ok text", language="en", source="whisper", is_generated=True
            )

    svc = TranscriptService([Bad(), Good()])
    assert svc.fetch("v").text == "ok text"

    with pytest.raises(TranscriptUnavailable):
        TranscriptService([Bad()]).fetch("v")
