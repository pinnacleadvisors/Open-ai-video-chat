from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from .config import Settings


class Engines:
    """Holds the heavy, loaded-once model objects shared across sessions.

    Per-session pipeline objects pull what they need from here. Anything
    cheap (queues, cancel events, state machines, history) lives on the
    session.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.whisper: Any = None
        self.vad: Any = None
        self.tts_backend: Any = None
        self.avatar_backend: Any = None
        self.http: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0)
        )
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._load_whisper()
        self._load_vad()
        self._load_tts()
        self._load_avatar()
        self._loaded = True
        logger.info("engines loaded")

    def _load_whisper(self) -> None:
        from faster_whisper import WhisperModel

        s = self.settings
        compute_type = s.stt_compute_type
        if s.device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        logger.info(f"loading faster-whisper {s.stt_model} on {s.device} ({compute_type})")
        self.whisper = WhisperModel(
            s.stt_model,
            device=s.device if s.device != "mps" else "cpu",
            compute_type=compute_type,
            download_root=str(s.models_dir / "whisper"),
        )

    def _load_vad(self) -> None:
        from silero_vad import load_silero_vad

        logger.info("loading silero VAD")
        self.vad = load_silero_vad(onnx=False)

    def _load_tts(self) -> None:
        from .pipeline.tts_backends import build_tts_backend

        self.tts_backend = build_tts_backend(self.settings)

    def _load_avatar(self) -> None:
        from .pipeline.avatar_backends import build_avatar_backend

        self.avatar_backend = build_avatar_backend(self.settings)

    async def close(self) -> None:
        await self.http.aclose()
