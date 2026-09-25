"""FastAPI voice endpoint tests + regression test for POST /chat.

Uses fake adapters patched into the voice_pipeline — no real Whisper or Edge TTS.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.voice.contracts import STTService, TTSService
from src.voice.pipeline import VoicePipeline


# ---------------------------------------------------------------------------
# Reusable fake adapters
# ---------------------------------------------------------------------------

class FakeSTT(STTService):
    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        return "I need help with my order"


class FakeFailSTT(STTService):
    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        raise RuntimeError("STT crashed")


class FakeTTS(TTSService):
    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        return b"\xff\xfb\x90\x00" + text.encode()[:20], "audio/mpeg"


class FakeFailTTS(TTSService):
    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        raise RuntimeError("TTS crashed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def voice_client():
    """TestClient with both text and voice pipelines mocked."""
    from src.api.server import app, pipeline, voice_pipeline

    # Mock text pipeline (same pattern as test_api.py)
    async def mock_init():
        pipeline.ready = True
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke = AsyncMock(return_value={
            "response_text": "Standard delivery takes 3-5 business days.",
            "sources": ["shipping.md"],
            "ticket_id": None,
        })
        pipeline.workflow = mock_workflow
        if hasattr(pipeline.retriever, 'search'):
            pipeline.retriever.search = AsyncMock(return_value=[])
            pipeline.retriever._store = MagicMock()

    # Swap voice adapters to fakes
    original_stt = voice_pipeline.stt
    original_tts = voice_pipeline.tts
    voice_pipeline.stt = FakeSTT()
    voice_pipeline.tts = FakeTTS()

    with patch.object(pipeline, 'initialize', side_effect=mock_init):
        with patch.object(voice_pipeline, 'initialize', new_callable=AsyncMock):
            with patch.object(voice_pipeline, 'cleanup', new_callable=AsyncMock):
                with TestClient(app, raise_server_exceptions=False) as c:
                    yield c, pipeline, voice_pipeline

    # Restore originals & clean up
    voice_pipeline.stt = original_stt
    voice_pipeline.tts = original_tts
    pipeline.sessions._sessions.clear()
    pipeline.tickets._tickets.clear()
    pipeline.tickets._session_ticket.clear()


# ---------------------------------------------------------------------------
# POST /voice/transcribe tests
# ---------------------------------------------------------------------------

def test_transcribe_success(voice_client):
    c, _, _ = voice_client
    resp = c.post(
        "/voice/transcribe",
        files={"file": ("test.wav", b"fake-audio-bytes", "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["transcript"] == "I need help with my order"
    assert "processing_time_ms" in data
    assert isinstance(data["processing_time_ms"], int)
    assert data["processing_time_ms"] >= 0


def test_transcribe_empty_audio(voice_client):
    c, _, _ = voice_client
    resp = c.post(
        "/voice/transcribe",
        files={"file": ("test.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 422


def test_transcribe_stt_failure(voice_client):
    c, _, vp = voice_client
    vp.stt = FakeFailSTT()
    resp = c.post(
        "/voice/transcribe",
        files={"file": ("test.wav", b"audio-data", "audio/wav")},
    )
    assert resp.status_code == 502
    data = resp.json()
    assert "detail" in data


# ---------------------------------------------------------------------------
# POST /voice/synthesize tests
# ---------------------------------------------------------------------------

def test_synthesize_success(voice_client):
    c, _, _ = voice_client
    resp = c.post(
        "/voice/synthesize",
        json={"message_id": "msg-0", "text": "Hello, how can I help?"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 0
    assert "x-processing-time-ms" in resp.headers


def test_synthesize_empty_text(voice_client):
    c, _, _ = voice_client
    resp = c.post(
        "/voice/synthesize",
        json={"message_id": "msg-0", "text": ""},
    )
    # Pydantic validation rejects empty text (min_length=1)
    assert resp.status_code == 422


def test_synthesize_tts_failure(voice_client):
    c, _, vp = voice_client
    vp.tts = FakeFailTTS()
    resp = c.post(
        "/voice/synthesize",
        json={"message_id": "msg-0", "text": "Some text"},
    )
    assert resp.status_code == 502
    data = resp.json()
    assert "detail" in data


def test_synthesize_correct_media_type(voice_client):
    c, _, _ = voice_client
    resp = c.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Test audio output"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"


# ---------------------------------------------------------------------------
# Message-audio association test (no cross-talk)
# ---------------------------------------------------------------------------

def test_distinct_messages_produce_distinct_audio(voice_client):
    c, _, _ = voice_client

    resp1 = c.post(
        "/voice/synthesize",
        json={"message_id": "msg-0", "text": "Response A"},
    )
    resp2 = c.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Response B"},
    )
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    # FakeTTS encodes the text into the audio, so they should differ
    assert resp1.content != resp2.content


# ---------------------------------------------------------------------------
# Regression: POST /chat still works after voice integration
# ---------------------------------------------------------------------------

def test_chat_still_works_after_voice_integration(voice_client):
    """POST /chat must be completely unchanged after adding voice endpoints."""
    c, pipeline, _ = voice_client
    resp = c.post("/chat", json={"session_id": "regression-1", "message": "shipping info"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "response" in data
    assert data["response"] != ""


def test_chat_validation_unchanged(voice_client):
    c, _, _ = voice_client
    resp = c.post("/chat", json={"session_id": "s1", "message": ""})
    assert resp.status_code == 422
    resp2 = c.post("/chat", json={"session_id": "", "message": "hello"})
    assert resp2.status_code == 422


def test_health_unchanged(voice_client):
    c, _, _ = voice_client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
