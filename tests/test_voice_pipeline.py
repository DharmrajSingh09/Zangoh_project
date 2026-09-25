"""Comprehensive voice pipeline and API tests.

All tests use fake adapters — no real model/network calls required.
The existing 36 text-agent tests remain untouched.
"""
from __future__ import annotations

import pytest

from src.voice.contracts import STTService, TTSService
from src.voice.pipeline import VoicePipeline


# ---------------------------------------------------------------------------
# Fake adapters (no real model calls)
# ---------------------------------------------------------------------------

class FakeSTT(STTService):
    """Always returns a deterministic transcript."""

    def __init__(self, transcript: str = "My payment was charged twice") -> None:
        self._transcript = transcript

    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        return self._transcript


class FakeFailingSTT(STTService):
    """Simulates an STT failure."""

    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        raise RuntimeError("STT model crashed")


class FakeEmptySTT(STTService):
    """Returns empty transcript (no understandable speech)."""

    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        return ""


class FakeTTS(TTSService):
    """Always returns deterministic fake audio bytes."""

    def __init__(self, audio: bytes = b"fake-audio-data", media_type: str = "audio/mpeg") -> None:
        self._audio = audio
        self._media_type = media_type

    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        return self._audio, self._media_type


class FakeFailingTTS(TTSService):
    """Simulates a TTS failure."""

    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        raise RuntimeError("TTS service unavailable")


class FakeEmptyTTS(TTSService):
    """Returns empty audio bytes."""

    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        return b"", "audio/mpeg"


# ---------------------------------------------------------------------------
# VoicePipeline unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_pipeline_contracts() -> None:
    """Original contract test — successful round-trip."""
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())

    transcript, _ = await pipeline.transcribe(b"fake-input", "audio/wav")
    audio, media_type, _ = await pipeline.synthesize("Agent response")

    assert transcript == "My payment was charged twice"
    assert audio == b"fake-audio-data"
    assert media_type == "audio/mpeg"


@pytest.mark.asyncio
async def test_transcribe_empty_audio_raises() -> None:
    """Empty audio bytes must be rejected before reaching the adapter."""
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    with pytest.raises(ValueError, match="Audio input is empty"):
        await pipeline.transcribe(b"", "audio/wav")


@pytest.mark.asyncio
async def test_transcribe_no_speech_detected() -> None:
    """Adapter returns empty string → pipeline raises ValueError."""
    pipeline = VoicePipeline(FakeEmptySTT(), FakeTTS())
    with pytest.raises(ValueError, match="No understandable speech"):
        await pipeline.transcribe(b"silent-audio", "audio/wav")


@pytest.mark.asyncio
async def test_transcribe_stt_failure() -> None:
    """Adapter raises RuntimeError → bubbles up."""
    pipeline = VoicePipeline(FakeFailingSTT(), FakeTTS())
    with pytest.raises(RuntimeError, match="STT model crashed"):
        await pipeline.transcribe(b"audio-data", "audio/wav")


@pytest.mark.asyncio
async def test_synthesize_empty_text_raises() -> None:
    """Empty/whitespace text must be rejected before reaching the adapter."""
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    with pytest.raises(ValueError, match="Text input is empty"):
        await pipeline.synthesize("   ")


@pytest.mark.asyncio
async def test_synthesize_tts_failure() -> None:
    """Adapter raises RuntimeError → bubbles up."""
    pipeline = VoicePipeline(FakeSTT(), FakeFailingTTS())
    with pytest.raises(RuntimeError, match="TTS service unavailable"):
        await pipeline.synthesize("Hello world")


@pytest.mark.asyncio
async def test_synthesize_empty_audio_raises() -> None:
    """Adapter returns empty bytes → pipeline raises ValueError."""
    pipeline = VoicePipeline(FakeSTT(), FakeEmptyTTS())
    with pytest.raises(ValueError, match="TTS returned empty audio"):
        await pipeline.synthesize("Hello world")


@pytest.mark.asyncio
async def test_transcribe_processing_time() -> None:
    """processing_time_ms must be non-negative."""
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    _, time_ms = await pipeline.transcribe(b"audio", "audio/wav")
    assert isinstance(time_ms, int)
    assert time_ms >= 0


@pytest.mark.asyncio
async def test_synthesize_processing_time_and_media_type() -> None:
    """processing_time_ms and media_type must be correct."""
    pipeline = VoicePipeline(FakeSTT(), FakeTTS(audio=b"mp3data", media_type="audio/mpeg"))
    audio, media_type, time_ms = await pipeline.synthesize("Hello")
    assert isinstance(time_ms, int)
    assert time_ms >= 0
    assert media_type == "audio/mpeg"
    assert audio == b"mp3data"


@pytest.mark.asyncio
async def test_message_audio_association() -> None:
    """Each response text produces its own distinct audio — no cross-talk."""
    stt = FakeSTT()
    tts_a = FakeTTS(audio=b"audio-for-msg-a")
    tts_b = FakeTTS(audio=b"audio-for-msg-b")

    pipe_a = VoicePipeline(stt, tts_a)
    pipe_b = VoicePipeline(stt, tts_b)

    audio_a, _, _ = await pipe_a.synthesize("Response A")
    audio_b, _, _ = await pipe_b.synthesize("Response B")

    assert audio_a == b"audio-for-msg-a"
    assert audio_b == b"audio-for-msg-b"
    assert audio_a != audio_b


@pytest.mark.asyncio
async def test_pipeline_initialize_and_cleanup() -> None:
    """Initialize and cleanup must not raise for well-behaved adapters."""
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    await pipeline.initialize()
    await pipeline.cleanup()
