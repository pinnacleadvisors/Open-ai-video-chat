from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucket:
    """Per-key token bucket. Thread-safe enough for asyncio (no awaits).

    Refills at `rate` tokens/sec up to `capacity`. Stores at most
    `max_keys` buckets (LRU eviction) so memory can't grow without bound.
    """

    def __init__(self, rate: float, capacity: float, max_keys: int = 10_000) -> None:
        self.rate = rate
        self.capacity = capacity
        self.max_keys = max_keys
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def allow(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self.capacity, updated=now)
            self._buckets[key] = bucket
            self._evict()
        else:
            elapsed = now - bucket.updated
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.rate)
            bucket.updated = now
            self._buckets.move_to_end(key)

        if bucket.tokens >= cost:
            bucket.tokens -= cost
            return True, 0.0
        deficit = cost - bucket.tokens
        retry = deficit / self.rate if self.rate > 0 else float("inf")
        return False, retry

    def _evict(self) -> None:
        while len(self._buckets) > self.max_keys:
            self._buckets.popitem(last=False)


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"


def enforce(bucket: TokenBucket, request: Request, cost: float = 1.0) -> None:
    """Raise HTTPException(429) if the IP is over its quota."""
    key = client_ip(request)
    allowed, retry = bucket.allow(key, cost=cost)
    if not allowed:
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(int(retry) + 1)})
