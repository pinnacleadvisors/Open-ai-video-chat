from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx
from loguru import logger

from ..config import Settings


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatState:
    system_prompt: str
    history: list[Message] = field(default_factory=list)
    max_history: int = 24

    def messages(self) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for m in self.history[-self.max_history :]:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    def add_user(self, text: str) -> None:
        self.history.append(Message("user", text))

    def add_assistant(self, text: str) -> None:
        if text:
            self.history.append(Message("assistant", text))


class LLM:
    """Streaming chat client. Supports Ollama and any OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = ChatState(system_prompt=settings.llm_system_prompt)
        self._cancel: Optional[asyncio.Event] = None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    def update_system_prompt(self, prompt: str) -> None:
        self.state.system_prompt = prompt

    def reset(self) -> None:
        self.state.history.clear()

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel.set()

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        """Stream the assistant reply token-by-token. Honors cancellation."""
        self.state.add_user(user_text)
        self._cancel = asyncio.Event()
        chunks: list[str] = []

        try:
            if self.settings.llm_backend == "ollama":
                stream = self._stream_ollama()
            else:
                stream = self._stream_openai()
            async for tok in stream:
                if self._cancel.is_set():
                    logger.info("llm reply cancelled (barge-in)")
                    break
                chunks.append(tok)
                yield tok
        finally:
            self.state.add_assistant("".join(chunks))
            self._cancel = None

    async def _stream_ollama(self) -> AsyncIterator[str]:
        import json

        url = self.settings.llm_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.settings.llm_model,
            "messages": self.state.messages(),
            "stream": True,
            "options": {
                "temperature": self.settings.llm_temperature,
                "num_predict": self.settings.llm_max_tokens,
            },
        }
        async with self._client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                msg = data.get("message") or {}
                content = msg.get("content") or ""
                if content:
                    yield content

    async def _stream_openai(self) -> AsyncIterator[str]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.settings.llm_api_key or "sk-dummy",
            base_url=self.settings.llm_base_url or None,
        )
        stream = await client.chat.completions.create(
            model=self.settings.llm_model,
            messages=self.state.messages(),
            stream=True,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def close(self) -> None:
        await self._client.aclose()
