from __future__ import annotations

import uuid

from loguru import logger
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = b"x-request-id"


class RequestIdMiddleware:
    """Reads or generates X-Request-ID; binds it to loguru contextvars and
    echoes it back on the response header so clients can correlate."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        rid_bytes = headers.get(REQUEST_ID_HEADER)
        rid = rid_bytes.decode("ascii", errors="ignore") if rid_bytes else uuid.uuid4().hex[:12]
        scope.setdefault("state", {})
        scope["state"]["request_id"] = rid

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers") or [])
                hdrs.append((REQUEST_ID_HEADER, rid.encode("ascii")))
                message["headers"] = hdrs
            await send(message)

        with logger.contextualize(request_id=rid):
            await self.app(scope, receive, send_with_id)


class SecurityHeadersMiddleware:
    """Adds baseline security headers. CSP intentionally omitted by default
    because the right policy is deploy-specific; set via reverse proxy."""

    HEADERS: tuple[tuple[bytes, bytes], ...] = (
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"permissions-policy", b"geolocation=(), microphone=(self), camera=(self)"),
    )

    def __init__(self, app: ASGIApp, hsts: bool = False) -> None:
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers") or [])
                existing = {k.lower() for k, _ in hdrs}
                for k, v in self.HEADERS:
                    if k not in existing:
                        hdrs.append((k, v))
                if self.hsts and b"strict-transport-security" not in existing:
                    hdrs.append((b"strict-transport-security", b"max-age=63072000; includeSubDomains"))
                message["headers"] = hdrs
            await send(message)

        await self.app(scope, receive, send_with_headers)
