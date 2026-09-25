"""Concrete STT adapter using faster-whisper (local, offline-capable)."""
from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path

from .contracts import STTService


class WhisperSTTService(STTService):
    """Speech-to-text using faster-whisper with a local Whisper model.

    Runs inference in a thread-pool executor to avoid blocking the async
    event loop. Accepts raw audio bytes with a media-type hint and writes
    them to a temporary file for the transcription engine.
    """

    # Media types that faster-whisper can handle (via ffmpeg internally)
    SUPPORTED_TYPES = frozenset({
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/webm", "audio/ogg", "audio/mpeg", "audio/mp3",
        "audio/mp4", "audio/flac", "audio/x-flac",
    })

    # Map media type to file extension for temp file
    _EXT_MAP: dict[str, str] = {
        "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
        "audio/webm": ".webm", "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
        "audio/mp4": ".mp4",
        "audio/flac": ".flac", "audio/x-flac": ".flac",
    }

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    async def initialize(self) -> None:
        """Load the whisper model in a background thread (heavy I/O)."""
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)

    def _load_model(self):
        from faster_whisper import WhisperModel
        return WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        """Transcribe audio bytes to text."""
        if self._model is None:
            raise RuntimeError("WhisperSTTService has not been initialized")

        normalized = media_type.lower().split(";")[0].strip()
        if normalized not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported audio type: {media_type}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_TYPES))}"
            )

        ext = self._EXT_MAP.get(normalized, ".wav")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe_sync, audio_bytes, ext
        )

    def _transcribe_sync(self, audio_bytes: bytes, ext: str) -> str:
        """Synchronous transcription for the thread pool."""
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, _ = self._model.transcribe(
                tmp_path,
                beam_size=5,
                language=None,  # auto-detect
                vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def cleanup(self) -> None:
        """Release model resources."""
        self._model = None
