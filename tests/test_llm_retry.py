import json

import httpx

from server.config import Settings
from server.pipeline.llm import LLMSession


def _settings(retries: int = 2) -> Settings:
    return Settings(
        device="cpu",
        llm_backend="ollama",
        llm_model="m",
        llm_base_url="http://fake",
        llm_max_retries=retries,
        llm_retry_backoff_s=0.0,
    )


def _ndjson_response(tokens: list[str]) -> bytes:
    lines = []
    for t in tokens:
        lines.append(json.dumps({"message": {"content": t}}))
    lines.append(json.dumps({"done": True}))
    return ("\n".join(lines) + "\n").encode()


async def _collect(session: LLMSession, text: str) -> str:
    out = []
    async for tok in session.stream_reply(text):
        out.append(tok)
    return "".join(out)


async def test_succeeds_on_first_attempt():
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=_ndjson_response(["he", "llo"]))
    )
    async with httpx.AsyncClient(transport=transport) as http:
        session = LLMSession(_settings(0), http, "s1")
        result = await _collect(session, "hi")
    assert result == "hello"


async def test_retries_on_transient_error_before_any_tokens():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, content=_ndjson_response(["ok"]))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        session = LLMSession(_settings(2), http, "s")
        result = await _collect(session, "hi")
    assert result == "ok"
    assert calls["n"] == 2


async def test_no_retry_left_returns_what_was_streamed():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=req)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        session = LLMSession(_settings(2), http, "s")
        result = await _collect(session, "hi")
    assert result == ""
    assert calls["n"] == 3  # 1 initial + 2 retries
