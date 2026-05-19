from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ..config import Settings

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


@dataclass
class Utterance:
    text: str
    language: str
    duration_s: float
    is_final: bool


class VADStateMachine:
    """Endpointing on top of a frame-level voicedness signal.

    Pure-Python, no model dependency — `voiced` is a function of the
    probability stream coming from Silero. Pulled out so it can be unit-tested.
    """

    def __init__(self, threshold: float, silence_ms: int):
        self.threshold = threshold
        self.silence_frames_needed = max(1, silence_ms // FRAME_MS)
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self.silence_streak = 0
        self.voiced_streak = 0

    def step(self, prob: float) -> tuple[bool, bool]:
        """Returns (is_voiced_now, just_ended_utterance)."""
        voiced = prob >= self.threshold
        ended = False
        if voiced:
            self.voiced_streak += 1
            self.silence_streak = 0
            if self.voiced_streak >= 2:
                self.speaking = True
        else:
            self.voiced_streak = 0
            if self.speaking:
                self.silence_streak += 1
                if self.silence_streak >= self.silence_frames_needed:
                    ended = True
                    self.speaking = False
                    self.silence_streak = 0
        return voiced, ended


class STTSession:
    """Per-conversation STT state.

    `engine.whisper` and `engine.vad` are loaded once at process start; this
    object only holds the streaming buffers + state machine + result queue.
    """

    def __init__(self, settings: Settings, whisper, vad_model, session_id: str):
        import torch

        self._torch = torch
        self.whisper = whisper
        self.vad_model = vad_model
        self.session_id = session_id
        self.language: str | None = None if settings.stt_language == "auto" else settings.stt_language
        self.vad = VADStateMachine(settings.vad_threshold, settings.vad_silence_ms)
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        self._utterance: list[np.ndarray] = []
        self._utterance_queue: asyncio.Queue[Utterance] = asyncio.Queue()
        self._voiced_event = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()

    @property
    def voiced(self) -> bool:
        return self.vad.speaking

    async def voiced_signal(self) -> None:
        await self._voiced_event.wait()
        self._voiced_event.clear()

    def push_pcm(self, pcm_f32: np.ndarray) -> None:
        if pcm_f32.ndim > 1:
            pcm_f32 = pcm_f32.mean(axis=-1)
        self._buffer.append(pcm_f32.astype(np.float32, copy=False))
        self._buffer_samples += pcm_f32.shape[0]
        while self._buffer_samples >= FRAME_SAMPLES:
            frame = self._take_frame()
            self._process_frame(frame)

    def _take_frame(self) -> np.ndarray:
        out = np.empty(FRAME_SAMPLES, dtype=np.float32)
        filled = 0
        while filled < FRAME_SAMPLES:
            head = self._buffer[0]
            need = FRAME_SAMPLES - filled
            if head.shape[0] <= need:
                out[filled : filled + head.shape[0]] = head
                filled += head.shape[0]
                self._buffer.popleft()
                self._buffer_samples -= head.shape[0]
            else:
                out[filled:] = head[:need]
                self._buffer[0] = head[need:]
                self._buffer_samples -= need
                filled = FRAME_SAMPLES
        return out

    def _process_frame(self, frame: np.ndarray) -> None:
        tensor = self._torch.from_numpy(frame)
        with self._torch.no_grad():
            prob = float(self.vad_model(tensor, SAMPLE_RATE).item())
        voiced, ended = self.vad.step(prob)

        if self.vad.speaking and not self._voiced_event.is_set():
            self._voiced_event.set()
        if self.vad.speaking or voiced:
            self._utterance.append(frame)
        if ended and self._utterance:
            audio = np.concatenate(self._utterance)
            self._utterance.clear()
            self._spawn(self._transcribe(audio))

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _transcribe(self, audio: np.ndarray) -> None:
        duration = audio.shape[0] / SAMPLE_RATE
        if duration < 0.3 or duration > 30.0:
            return
        loop = asyncio.get_running_loop()
        try:
            text, lang = await loop.run_in_executor(None, self._run_whisper, audio)
        except Exception as e:
            logger.bind(session=self.session_id).exception(f"whisper failed: {e}")
            return
        text = text.strip()
        if not text:
            return
        await self._utterance_queue.put(
            Utterance(text=text, language=lang, duration_s=duration, is_final=True)
        )

    def _run_whisper(self, audio: np.ndarray) -> tuple[str, str]:
        segments, info = self.whisper.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = "".join(seg.text for seg in segments)
        return text, info.language

    async def stream(self) -> AsyncIterator[Utterance]:
        while True:
            yield await self._utterance_queue.get()

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        for t in list(self._tasks):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
