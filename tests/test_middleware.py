from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from server.middleware import RequestIdMiddleware, SecurityHeadersMiddleware


def _app(middleware) -> FastAPI:
    app = FastAPI()
    app.add_middleware(middleware)

    @app.get("/")
    def root():
        return JSONResponse({"ok": True})

    return app


def test_security_headers_added():
    client = TestClient(_app(SecurityHeadersMiddleware))
    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["referrer-policy"].startswith("strict-origin")
    assert "permissions-policy" in r.headers
    assert "strict-transport-security" not in r.headers  # HSTS off by default


def test_security_headers_hsts_when_enabled():
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, hsts=True)

    @app.get("/")
    def root():
        return {"ok": True}

    client = TestClient(app)
    r = client.get("/")
    assert "strict-transport-security" in r.headers


def test_request_id_generated_when_not_provided():
    client = TestClient(_app(RequestIdMiddleware))
    r = client.get("/")
    rid = r.headers.get("x-request-id")
    assert rid
    assert len(rid) >= 8


def test_request_id_echoed_when_provided():
    client = TestClient(_app(RequestIdMiddleware))
    r = client.get("/", headers={"X-Request-ID": "test-abc-123"})
    assert r.headers["x-request-id"] == "test-abc-123"


def test_request_ids_are_unique_per_request():
    client = TestClient(_app(RequestIdMiddleware))
    ids = {client.get("/").headers["x-request-id"] for _ in range(5)}
    assert len(ids) == 5
