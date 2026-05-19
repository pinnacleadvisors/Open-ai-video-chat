from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable

from loguru import logger


async def graceful_drain(
    close_sessions: Callable[[], Awaitable[None]],
    *,
    timeout: float = 15.0,
) -> None:
    """Close all live sessions, then return. Bounded by `timeout`.

    Call this from your lifespan shutdown hook before tearing down engines.
    The shutdown is bounded so a wedged session can't block process exit
    forever — uvicorn's own timeout-graceful-shutdown still applies on top.
    """
    try:
        await asyncio.wait_for(close_sessions(), timeout=timeout)
    except TimeoutError:
        logger.warning(f"graceful drain timed out after {timeout}s; sessions may be force-closed")


def install_signal_handlers(loop: asyncio.AbstractEventLoop, on_signal: Callable[[], None]) -> None:
    """Register SIGTERM/SIGINT handlers that trigger `on_signal` exactly once.

    On systems that don't support loop.add_signal_handler (Windows), this
    is a no-op — uvicorn's own handlers still run.
    """
    fired = False

    def _handler() -> None:
        nonlocal fired
        if fired:
            return
        fired = True
        logger.info("shutdown signal received, draining")
        try:
            on_signal()
        except Exception as e:
            logger.exception(f"shutdown handler raised: {e}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except NotImplementedError:
            return
