from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ..config import Settings
from ..metrics import metrics
from ..utils.fanout import Broadcaster
from .avatar import AvatarSession
from .llm import LLMSession
from .stt import STTSession
from .tts import TTSSession


@dataclass
class TranscriptEvent:
    role: str   # "user" | "assistant"
    text: str
    final: bool


class Orchestrator:
    """Drives STT → LLM → TTS → Avatar for a single conversation with barge-in."""

    def __init__(
        self,
        settings: Settings,
        stt: STTSession,
        llm: LLMSession,
        tts: TTSSession,
        avatar: AvatarSession,
        session_id: str,
    ):
        self.settings = settings
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.avatar = avatar
        self.session_id = session_id
        self.transcripts: Broadcaster[TranscriptEvent] = Broadcaster(maxsize=64)
        self._active_reply: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._log = logger.bind(session=session_id)

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self._consume_user_utterances(), name=f"orch.user.{self.session_id}"))
        if self.settings.barge_in:
            self._tasks.append(asyncio.create_task(self._watch_barge_in(), name=f"orch.barge.{self.session_id}"))
        metrics.sessions_active.inc()

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        if self._active_reply and not self._active_reply.done():
            self._active_reply.cancel()
            try:
                await self._active_reply
            except (asyncio.CancelledError, Exception):
                pass
        await self.transcripts.close()
        await self.avatar.close()
        await self.stt.close()
        metrics.sessions_active.dec()

    def push_audio(self, pcm_f32: np.ndarray) -> None:
        self.stt.push_pcm(pcm_f32)

    async def _consume_user_utterances(self) -> None:
        async for utt in self.stt.stream():
            self._log.info(f"user: {utt.text}")
            metrics.utterances_total.inc()
            await self.transcripts.publish(TranscriptEvent("user", utt.text, True))
            if self._active_reply and not self._active_reply.done():
                self._interrupt()
                try:
                    await self._active_reply
                except (asyncio.CancelledError, Exception):
                    pass
            self._active_reply = asyncio.create_task(
                self._reply(utt.text), name=f"reply.{self.session_id}"
            )

    async def _watch_barge_in(self) -> None:
        while True:
            await self.stt.voiced_signal()
            if self._active_reply and not self._active_reply.done():
                self._log.info("barge-in detected")
                metrics.barge_ins_total.inc()
                self._interrupt()

    def _interrupt(self) -> None:
        self.llm.cancel()
        self.tts.cancel()
        self.avatar.cancel()
        if self._active_reply:
            self._active_reply.cancel()

    async def _reply(self, user_text: str) -> None:
        self.tts.reset()
        self.avatar.reset()

        producer = asyncio.create_task(self._produce_speech(user_text), name="reply.produce")
        consumer = asyncio.create_task(self._render_speech(), name="reply.render")

        try:
            await asyncio.gather(producer, consumer)
        except asyncio.CancelledError:
            producer.cancel()
            consumer.cancel()
            raise

    async def _produce_speech(self, user_text: str) -> None:
        full = ""
        t0 = time.monotonic()
        first_token = None
        async for tok in self.llm.stream_reply(user_text):
            if first_token is None:
                first_token = time.monotonic() - t0
                metrics.llm_ttft_seconds.observe(first_token)
            full += tok
            await self.transcripts.publish(TranscriptEvent("assistant", tok, False))
            await self.tts.feed_token(tok)
        await self.tts.flush()
        await self.transcripts.publish(TranscriptEvent("assistant", full, True))
        self._log.info(f"assistant: {full}")

    async def _render_speech(self) -> None:
        async for chunk in self.tts.stream():
            if chunk.pcm.size == 0:
                continue
            t0 = time.monotonic()
            await self.avatar.render(chunk.pcm, chunk.sample_rate)
            metrics.avatar_render_seconds.observe(time.monotonic() - t0)
