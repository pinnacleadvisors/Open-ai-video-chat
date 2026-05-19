from __future__ import annotations

import hmac
from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp

_PUBLIC_PATHS = {"/api/health", "/api/ready", "/api/metrics"}

TokenSource = str | Callable[[], str]


class BearerTokenMiddleware:
    """Optional bearer-token auth. If the resolved token is empty, no-op.

    Accepts either a static string or a zero-arg callable so the token can
    be rotated at runtime (config reload) without rebuilding the middleware.
    """

    def __init__(self, app: ASGIApp, token: TokenSource) -> None:
        self.app = app
        self._token = token

    def _current_token(self) -> str:
        return self._token() if callable(self._token) else self._token

    async def __call__(self, scope, receive, send) -> None:
        token = self._current_token()
        if not token or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        if not self._authorized(scope, token):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    def _authorized(scope, token: str) -> bool:
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1", errors="ignore")
        if auth.lower().startswith("bearer "):
            return hmac.compare_digest(auth[7:].strip(), token)
        qs = dict(
            pair.split("=", 1) if "=" in pair else (pair, "")
            for pair in scope.get("query_string", b"").decode("latin-1").split("&")
            if pair
        )
        return hmac.compare_digest(qs.get("token", ""), token)
