from __future__ import annotations

import hmac

from starlette.responses import JSONResponse
from starlette.types import ASGIApp

_PUBLIC_PATHS = {"/api/health", "/api/ready", "/api/metrics"}


class BearerTokenMiddleware:
    """Optional bearer-token auth. If `token` is empty, the middleware is a no-op."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        if not self.token or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        if not self._authorized(scope):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    def _authorized(self, scope) -> bool:  # type: ignore[no-untyped-def]
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1", errors="ignore")
        if auth.lower().startswith("bearer "):
            return hmac.compare_digest(auth[7:].strip(), self.token)
        # Allow token in query string for websockets (most browsers can't set headers there).
        qs = dict(
            pair.split("=", 1) if "=" in pair else (pair, "")
            for pair in scope.get("query_string", b"").decode("latin-1").split("&")
            if pair
        )
        return hmac.compare_digest(qs.get("token", ""), self.token)
