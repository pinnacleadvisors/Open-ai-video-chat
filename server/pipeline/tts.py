from __future__ import annotations

import asyncio
import io
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from ..config import Settings


# Split incoming token stream into "speakable" chunks so the avatar can
# start talking before the LLM finishes generating.
_SENTENCE_BOUNDARY = re.compile(r"([\.!\?…]+|,(?=\s)|;|:)")


@dataclass
class AudioChunk:
    pcm: np.ndarray      # float32, mono, settings.tts_sample_rate
    sample_rate: int
    text: str
    is_final: bool


class _Phraser:
    """Buffer LLM tokens and emit speakable phrases."""

    MIN_CHARS = 24
    MAX_CHARS = 180

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        self._buf += token
        out: list[str] = []
        while True:
            phrase = self._cut()
            if phrase is None:
                break
            out.append(phrase)
        return out

    def flush(self) -> Optional[str]:
        if self._buf.strip():
            phrase, self._buf = self._buf, ""
            return phrase.strip()
        return None

    def _cut(self) -> Optional[str]:
        if len(self._buf) >= self.MAX_CHARS:
            phrase, self._buf = self._buf, ""
            return phrase.strip()
        m = None
        for match in _SENTENCE_BOUNDARY.finditer(self._buf):
            m = match
            if match.end() >= self.MIN_CHARS:
                break
        if m is None or m.end() < self.MIN_CHARS:
            return None
        idx = m.end()
        phrase, self._buf = self._buf[:idx], self._buf[idx:]
        return phrase.strip()


class _PiperBackend:
    """Piper TTS — fast, runs well on CPU. Uses the piper CLI under the hood."""

    def __init__(self, settings: Settings):
        from piper import PiperVoice

        model_path = self._resolve_voice(settings)
        logger.info(f"loading piper voice {model_path}")
        self.voice = PiperVoice.load(str(model_path))
        self.sample_rate = self.voice.config.sample_rate

    def _resolve_voice(self, settings: Settings) -> Path:
        voices_dir = settings.models_dir / "piper"
        candidate = voices_dir / f"{settings.tts_voice}.onnx"
        if not candidate.exists():
            raise FileNotFoundError(
                f"piper voice {candidate} not found. "
                f"Run scripts/setup.sh or scripts/download_models.py to fetch it."
            )
        return candidate

    def synth(self, text: str) -> np.ndarray:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if sr != self.sample_rate:
            self.sample_rate = sr
        return pcm


class _XTTSBackend:
    """Coqui XTTS-v2 — supports voice cloning from a 6-30s reference clip."""

    def __init__(self, settings: Settings):
        from TTS.api import TTS as CoquiTTS

        self.settings = settings
        device = settings.device if settings.device != "mps" else "cpu"
        logger.info(f"loading XTTS-v2 on {device}")
        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.sample_rate = 24000
        self.speaker_wav = settings.xtts_speaker_wav or None

    def synth(self, text: str) -> np.ndarray:
        wav = self.tts.tts(
            text=text,
            speaker_wav=self.speaker_wav,
            language="en",
            split_sentences=False,
        )
        return np.asarray(wav, dtype=np.float32)


class TTS:
    """Streaming TTS. Feed LLM tokens via `feed_token`, get AudioChunk via `stream`."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.tts_backend == "xtts":
            self.backend = _XTTSBackend(settings)
        else:
            self.backend = _PiperBackend(settings)
        self.sample_rate = self.backend.sample_rate
        self._phraser = _Phraser()
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
        self._cancel = asyncio.Event()

    def reset(self) -> None:
        self._phraser = _Phraser()
        self._cancel.clear()

    def cancel(self) -> None:
        self._cancel.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def feed_token(self, token: str) -> None:
        phrases = self._phraser.feed(token)
        for phrase in phrases:
            await self._synth(phrase, is_final=False)

    async def flush(self) -> None:
        tail = self._phraser.flush()
        if tail:
            await self._synth(tail, is_final=True)
        else:
            await self._queue.put(AudioChunk(np.zeros(0, dtype=np.float32), self.sample_rate, "", True))

    async def _synth(self, text: str, is_final: bool) -> None:
        if self._cancel.is_set():
            return
        loop = asyncio.get_running_loop()
        try:
            pcm = await loop.run_in_executor(None, self.backend.synth, text)
        except Exception as e:
            logger.exception(f"tts failed for {text!r}: {e}")
            return
        if self._cancel.is_set():
            return
        await self._queue.put(AudioChunk(pcm=pcm, sample_rate=self.backend.sample_rate, text=text, is_final=is_final))

    async def stream(self) -> AsyncIterator[AudioChunk]:
        while True:
            chunk = await self._queue.get()
            yield chunk
            if chunk.is_final:
                return
