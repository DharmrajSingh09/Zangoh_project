"""Concrete TTS adapter using edge-tts (free, no API key required)."""
from __future__ import annotations

import asyncio
import io

from .contracts import TTSService


class EdgeTTSService(TTSService):
    """Text-to-speech using Microsoft Edge TTS (free, no auth).

    Edge TTS is a free cloud service accessible without an API key.
    It outputs MP3 audio and supports many voices/languages.
    """

    DEFAULT_VOICE = "en-US-AriaNeural"

    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice or self.DEFAULT_VOICE

    async def initialize(self) -> None:
        """Verify the edge_tts package is available."""
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Run: pip install edge-tts"
            ) from exc

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        """Convert text to MP3 audio bytes."""
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_bytes = audio_buffer.getvalue()
        if not audio_bytes:
            raise RuntimeError("Edge TTS returned no audio data")

        return audio_bytes, "audio/mpeg"

    async def cleanup(self) -> None:
        """No persistent resources to release."""
        pass
