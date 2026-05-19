from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Broadcaster(Generic[T]):
    """Single-producer, many-subscriber asynchronous broadcaster.

    Each subscriber gets its own bounded queue. On overflow, the oldest
    item in that subscriber's queue is dropped — slow consumers can't
    stall the pipeline. `close()` signals end-of-stream without losing
    pending items.
    """

    def __init__(self, maxsize: int = 8) -> None:
        self._maxsize = maxsize
        self._subs: list[_Subscription[T]] = []
        self._closed = False

    def subscribe(self) -> _Subscription[T]:
        sub: _Subscription[T] = _Subscription(self, self._maxsize)
        self._subs.append(sub)
        if self._closed:
            sub._close()
        return sub

    def _unsubscribe(self, sub: _Subscription[T]) -> None:
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    async def publish(self, item: T) -> None:
        if self._closed:
            return
        for sub in list(self._subs):
            sub._offer(item)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sub in list(self._subs):
            sub._close()

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


class _Subscription(Generic[T]):
    def __init__(self, parent: Broadcaster[T], maxsize: int):
        self._parent = parent
        self._q: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._closed = asyncio.Event()

    def _offer(self, item: T) -> None:
        if self._q.full():
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def _close(self) -> None:
        self._closed.set()

    def __aiter__(self) -> AsyncIterator[T]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[T]:
        try:
            while True:
                if self._q.empty() and self._closed.is_set():
                    return
                # Wait for either a new item or close, whichever comes first.
                get = asyncio.create_task(self._q.get())
                closed = asyncio.create_task(self._closed.wait())
                done, pending = await asyncio.wait(
                    {get, closed}, return_when=asyncio.FIRST_COMPLETED
                )
                if get in done:
                    closed.cancel()
                    yield get.result()
                else:
                    get.cancel()
                    # Drain anything still queued, then exit.
                    while not self._q.empty():
                        yield self._q.get_nowait()
                    return
        finally:
            self._parent._unsubscribe(self)

    async def aclose(self) -> None:
        self._close()
        self._parent._unsubscribe(self)
