import pytest
from fastapi import HTTPException
from starlette.requests import Request

from server.rate_limit import TokenBucket, client_ip, enforce


def _fake_request(ip: str = "1.2.3.4", headers: dict | None = None) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (ip, 0),
        "method": "GET",
        "path": "/",
    }
    return Request(scope)


def test_token_bucket_allows_within_capacity():
    b = TokenBucket(rate=10, capacity=5)
    for _ in range(5):
        allowed, _ = b.allow("ip1")
        assert allowed


def test_token_bucket_blocks_over_capacity():
    b = TokenBucket(rate=10, capacity=3)
    for _ in range(3):
        assert b.allow("ip1")[0]
    allowed, retry = b.allow("ip1")
    assert not allowed
    assert retry > 0


def test_token_bucket_refills_over_time(monkeypatch):
    fake_t = [0.0]
    monkeypatch.setattr("server.rate_limit.time.monotonic", lambda: fake_t[0])
    b = TokenBucket(rate=2.0, capacity=2)
    assert b.allow("ip")[0]
    assert b.allow("ip")[0]
    assert not b.allow("ip")[0]
    fake_t[0] += 1.0  # 2 tokens refilled
    assert b.allow("ip")[0]
    assert b.allow("ip")[0]
    assert not b.allow("ip")[0]


def test_token_bucket_independent_keys():
    b = TokenBucket(rate=10, capacity=1)
    assert b.allow("a")[0]
    assert not b.allow("a")[0]
    assert b.allow("b")[0]


def test_token_bucket_lru_eviction():
    b = TokenBucket(rate=10, capacity=1, max_keys=3)
    for k in ["a", "b", "c", "d"]:
        b.allow(k)
    assert "a" not in b._buckets  # evicted
    assert "d" in b._buckets


def test_client_ip_from_x_forwarded_for():
    req = _fake_request(headers={"x-forwarded-for": "10.0.0.1, 192.168.1.1"})
    assert client_ip(req) == "10.0.0.1"


def test_client_ip_from_x_real_ip_fallback():
    req = _fake_request(headers={"x-real-ip": "10.0.0.2"})
    assert client_ip(req) == "10.0.0.2"


def test_client_ip_from_socket_when_no_proxy_header():
    req = _fake_request(ip="5.6.7.8")
    assert client_ip(req) == "5.6.7.8"


def test_enforce_raises_429_when_over_limit():
    b = TokenBucket(rate=1, capacity=1)
    req = _fake_request(ip="1.2.3.4")
    enforce(b, req)
    with pytest.raises(HTTPException) as ei:
        enforce(b, req)
    assert ei.value.status_code == 429
    assert "Retry-After" in ei.value.headers
