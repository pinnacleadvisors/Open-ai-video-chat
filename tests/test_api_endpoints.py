"""Integration tests over the FastAPI app with engines stubbed.

We set OAVC_SKIP_ENGINES=1 in conftest so lifespan doesn't actually load
heavy ML models. We then inject a fake engines object so endpoints that
need it (like /api/voices) still work.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OAVC_SKIP_ENGINES", "1")
    monkeypatch.setenv("AUTH_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_OFFER_PER_MIN", "60")
    monkeypatch.setenv("RATE_LIMIT_UPLOAD_PER_MIN", "60")
    # Sandbox writes
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    from server import main as app_main
    from server.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    with TestClient(app_main.app) as tc:
        # Stub the engines so /api/voices works without actually loading models.
        app_main.state.engines.tts_backend = SimpleNamespace(
            available_voices=lambda: ["fake-voice-1", "fake-voice-2"],
            set_voice=lambda _v: None,
            speaker_wav=None,
        )
        yield tc

    get_settings.cache_clear()  # type: ignore[attr-defined]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "sessions" in body


def test_metrics_returns_prometheus_format(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "oavc_sessions_active" in r.text


def test_ready_returns_check_breakdown(client):
    r = client.get("/api/ready")
    # Without real engines and weights this is 503 — that's the correct signal.
    assert r.status_code in (200, 503)
    body = r.json()
    assert "checks" in body
    names = {c["name"] for c in body["checks"]}
    assert "engines" in names


def test_security_headers_present(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"


def test_request_id_round_trip(client):
    r = client.get("/api/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"


def test_voices_listing(client):
    r = client.get("/api/voices")
    assert r.status_code == 200
    assert r.json()["voices"] == ["fake-voice-1", "fake-voice-2"]


def test_personas_crud(client):
    # initially empty
    r = client.get("/api/personas")
    assert r.status_code == 200
    assert r.json() == []

    # create
    img = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    r = client.post(
        "/api/personas",
        params={"name": "Alice", "voice": "fake-voice-1"},
        files={"file": ("a.png", img, "image/png")},
    )
    assert r.status_code == 200, r.text
    persona = r.json()
    assert persona["name"] == "Alice"

    # list returns it
    r = client.get("/api/personas")
    assert any(p["id"] == persona["id"] for p in r.json())

    # delete
    r = client.delete(f"/api/personas/{persona['id']}")
    assert r.json()["deleted"] is True


def test_persona_create_rejects_non_image(client):
    r = client.post(
        "/api/personas",
        params={"name": "X", "voice": "v"},
        files={"file": ("a.bin", b"\x00\x01\x02not an image", "application/octet-stream")},
    )
    assert r.status_code == 415


def test_auth_blocks_when_token_set(tmp_path, monkeypatch):
    monkeypatch.setenv("OAVC_SKIP_ENGINES", "1")
    monkeypatch.setenv("AUTH_TOKEN", "secret123")
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    from server import main as app_main
    from server.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    with TestClient(app_main.app) as tc:
        # Public endpoints still allowed
        assert tc.get("/api/health").status_code == 200
        # Personas requires auth
        assert tc.get("/api/personas").status_code == 401
        # With token works
        r = tc.get("/api/personas", headers={"Authorization": "Bearer secret123"})
        assert r.status_code == 200
    get_settings.cache_clear()  # type: ignore[attr-defined]


def test_rate_limit_kicks_in(tmp_path, monkeypatch):
    monkeypatch.setenv("OAVC_SKIP_ENGINES", "1")
    monkeypatch.setenv("AUTH_TOKEN", "")
    monkeypatch.setenv("RATE_LIMIT_UPLOAD_PER_MIN", "2")
    monkeypatch.setenv("MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    from server import main as app_main
    from server.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    with TestClient(app_main.app) as tc:
        img = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        # capacity = 2 → first two pass, third gets 429
        for _ in range(2):
            assert tc.post(
                "/api/personas",
                params={"name": "X", "voice": "v"},
                files={"file": ("a.png", img, "image/png")},
            ).status_code == 200
        r = tc.post(
            "/api/personas",
            params={"name": "X", "voice": "v"},
            files={"file": ("a.png", img, "image/png")},
        )
        assert r.status_code == 429
        assert "Retry-After" in r.headers
    get_settings.cache_clear()  # type: ignore[attr-defined]
