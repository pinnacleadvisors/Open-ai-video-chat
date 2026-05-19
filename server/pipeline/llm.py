from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
from loguru import logger

from ..config import Settings


@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatState:
    """Conversation history with a system prompt. Pure data — no I/O."""

    system_prompt: str
    history: list[Message] = field(default_factory=list)
    max_history: int = 24

    def messages(self) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for m in self.history[-self.max_history :]:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    def add_user(self, text: str) -> None:
        text = text.strip()
        if text:
            self.history.append(Message("user", text))

    def add_assistant(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.history.append(Message("assistant", text))

    def reset(self) -> None:
        self.history.clear()


class LLMSession:
    """Per-conversation streaming chat client. Shares the http client via engine."""

    def __init__(self, settings: Settings, http: httpx.AsyncClient, session_id: str):
        self.settings = settings
        self.http = http
        self.session_id = session_id
        self.state = ChatState(system_prompt=settings.llm_system_prompt)
        self._cancel: asyncio.Event | None = None

    def update_system_prompt(self, prompt: str) -> None:
        self.state.system_prompt = prompt

    def reset(self) -> None:
        self.state.reset()

    def cancel(self) -> None:
        if self._cancel is not None:
            self._cancel.set()

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        self.state.add_user(user_text)
        self._cancel = asyncio.Event()
        chunks: list[str] = []
        log = logger.bind(session=self.session_id)

        attempts = max(1, self.settings.llm_max_retries + 1)
        backoff = self.settings.llm_retry_backoff_s
        try:
            for attempt in range(1, attempts + 1):
                try:
                    stream = (
                        self._stream_ollama() if self.settings.llm_backend == "ollama"
                        else self._stream_openai()
                    )
                    async for tok in stream:
                        if self._cancel.is_set():
                            log.info("llm reply cancelled")
                            return
                        chunks.append(tok)
                        yield tok
                    return
                except httpx.HTTPError as e:
                    # Only safe to retry if we haven't emitted anything yet.
                    if chunks or attempt == attempts:
                        log.warning(f"llm stream failed after {len(chunks)} tokens: {e}")
                        return
                    delay = backoff * (2 ** (attempt - 1))
                    log.warning(f"llm stream attempt {attempt} failed ({e}); retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                except Exception as e:
                    log.exception(f"llm stream errored: {e}")
                    return
        finally:
            self.state.add_assistant("".join(chunks))
            self._cancel = None

    async def _stream_ollama(self) -> AsyncIterator[str]:
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
        async with self.http.stream("POST", url, json=payload) as r:
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
                content = (data.get("message") or {}).get("content") or ""
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
