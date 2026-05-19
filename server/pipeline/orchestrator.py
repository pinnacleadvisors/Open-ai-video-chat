from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

import numpy as np
from loguru import logger

from ..config import Settings
from .stt import STT
from .llm import LLM
from .tts import TTS, AudioChunk
from .avatar import Avatar, AVPair


@dataclass
class TranscriptEvent:
    role: str       # "user" | "assistant"
    text: str
    final: bool


class Orchestrator:
    """Drives the full STT → LLM → TTS → Avatar pipeline with barge-in.

    A single instance maps to a single conversation. Push raw mic audio
    in via `push_audio`, await `av_stream()` to get rendered AVPairs,
    await `transcripts()` for text events.
    """

    def __init__(
        self,
        settings: Settings,
        stt: STT,
        llm: LLM,
        tts: TTS,
        avatar: Avatar,
    ):
        self.settings = settings
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.avatar = avatar
        self._transcripts: asyncio.Queue[TranscriptEvent] = asyncio.Queue()
        self._active_reply: Optional[asyncio.Task] = None
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self._consume_user_utterances(), name="orchestrator.user"))
        if self.settings.barge_in:
            self._tasks.append(asyncio.create_task(self._watch_for_barge_in(), name="orchestrator.barge"))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    def push_audio(self, pcm_f32: np.ndarray) -> None:
        self.stt.push_pcm(pcm_f32)

    async def transcripts(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            yield await self._transcripts.get()

    async def av_stream(self) -> AsyncIterator[AVPair]:
        async for pair in self.avatar.stream():
            yield pair

    async def _consume_user_utterances(self) -> None:
        async for utt in self.stt.stream():
            logger.info(f"user: {utt.text}")
            await self._transcripts.put(TranscriptEvent("user", utt.text, True))
            if self._active_reply and not self._active_reply.done():
                self._interrupt()
                try:
                    await self._active_reply
                except asyncio.CancelledError:
                    pass
            self._active_reply = asyncio.create_task(self._reply(utt.text), name="orchestrator.reply")

    async def _watch_for_barge_in(self) -> None:
        while True:
            await self.stt.voiced_signal()
            if self._active_reply and not self._active_reply.done():
                logger.info("barge-in detected, cancelling assistant turn")
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
        async for tok in self.llm.stream_reply(user_text):
            full += tok
            await self._transcripts.put(TranscriptEvent("assistant", tok, False))
            await self.tts.feed_token(tok)
        await self.tts.flush()
        await self._transcripts.put(TranscriptEvent("assistant", full, True))
        logger.info(f"assistant: {full}")

    async def _render_speech(self) -> None:
        async for chunk in self.tts.stream():
            if chunk.pcm.size == 0:
                continue
            await self.avatar.render(chunk.pcm, chunk.sample_rate)
