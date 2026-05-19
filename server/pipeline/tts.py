from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ..config import Settings
from .tts_backends import TTSBackend

_SENTENCE_BOUNDARY = re.compile(r"([\.!\?…]+|,(?=\s)|;|:)")


@dataclass
class AudioChunk:
    pcm: np.ndarray
    sample_rate: int
    text: str
    is_final: bool


class Phraser:
    """Buffer LLM tokens and emit speakable phrases.

    Pure logic; no I/O. Sized so that the first phrase comes out fast (low
    time-to-first-audio) and later phrases are larger (better TTS prosody).
    """

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

    def flush(self) -> str | None:
        if self._buf.strip():
            phrase, self._buf = self._buf, ""
            return phrase.strip()
        return None

    def _cut(self) -> str | None:
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


class TTSSession:
    """Per-conversation TTS state. Shares the loaded voice/model via the backend."""

    def __init__(self, settings: Settings, backend: TTSBackend, session_id: str):
        self.settings = settings
        self.backend = backend
        self.session_id = session_id
        self.sample_rate = backend.sample_rate
        self._phraser = Phraser()
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
        self._cancel = asyncio.Event()

    def reset(self) -> None:
        self._phraser = Phraser()
        self._cancel.clear()

    def cancel(self) -> None:
        self._cancel.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def feed_token(self, token: str) -> None:
        for phrase in self._phraser.feed(token):
            await self._synth(phrase, is_final=False)

    async def flush(self) -> None:
        tail = self._phraser.flush()
        if tail:
            await self._synth(tail, is_final=True)
        else:
            await self._queue.put(
                AudioChunk(np.zeros(0, dtype=np.float32), self.backend.sample_rate, "", True)
            )

    async def _synth(self, text: str, is_final: bool) -> None:
        if self._cancel.is_set():
            return
        loop = asyncio.get_running_loop()
        try:
            pcm = await loop.run_in_executor(None, self.backend.synth, text)
        except Exception as e:
            logger.bind(session=self.session_id).exception(f"tts failed for {text!r}: {e}")
            return
        if self._cancel.is_set():
            return
        await self._queue.put(
            AudioChunk(
                pcm=pcm,
                sample_rate=self.backend.sample_rate,
                text=text,
                is_final=is_final,
            )
        )

    async def stream(self) -> AsyncIterator[AudioChunk]:
        while True:
            chunk = await self._queue.get()
            yield chunk
            if chunk.is_final:
                return
