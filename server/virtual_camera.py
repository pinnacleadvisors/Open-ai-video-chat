from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from .config import Settings

if TYPE_CHECKING:
    from .pipeline import Orchestrator


class VirtualCameraPublisher:
    """Publishes one Orchestrator's avatar frames to a v4l2loopback / OBS virtual cam.

    Multi-session deployments should run a separate process per session if
    multiple virtual cameras are needed; in this single-process build, the
    virtual camera follows whichever orchestrator is `attach`-ed (typically
    the first / primary session).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._orch: Orchestrator | None = None
        self._sub = None
        self._pending: list[np.ndarray] = []

    def attach(self, orch: Orchestrator) -> None:
        """Attach (or re-attach) to an orchestrator's avatar stream."""
        if not self.settings.enable_virtual_camera:
            return
        self._orch = orch
        self._sub = orch.avatar.frames.subscribe()
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="virtualcam")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
        if self._sub:
            await self._sub.aclose()

    async def _run(self) -> None:
        import pyvirtualcam

        w = self.settings.virtual_camera_width
        h = self.settings.virtual_camera_height
        fps = self.settings.lipsync_fps
        try:
            cam = pyvirtualcam.Camera(
                width=w, height=h, fps=fps,
                device=self.settings.virtual_camera_device,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
        except Exception as e:
            logger.error(
                f"virtual camera unavailable ({e}). On linux: "
                '`sudo modprobe v4l2loopback devices=1 card_label="OpenAI Video Chat"`'
            )
            return

        logger.info(f"virtual camera publishing to {cam.device} at {w}x{h}@{fps}fps")
        consume_task = asyncio.create_task(self._consume(), name="virtualcam.consume")
        try:
            while not self._stop.is_set():
                if self._pending:
                    frame = self._pending.pop(0)
                else:
                    idle = self._orch.avatar.idle_frame() if self._orch else None
                    frame = idle if idle is not None else np.zeros((h, w, 3), dtype=np.uint8)
                cam.send(frame)
                cam.sleep_until_next_frame()
                await asyncio.sleep(0)
        finally:
            consume_task.cancel()
            try:
                await consume_task
            except (asyncio.CancelledError, Exception):
                pass
            cam.close()

    async def _consume(self) -> None:
        if self._sub is None:
            return
        async for pair in self._sub:
            for vf in pair.video:
                self._pending.append(vf.image)
