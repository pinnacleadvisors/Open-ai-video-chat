from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
from loguru import logger

from .config import Settings
from .pipeline import Orchestrator


class VirtualCameraPublisher:
    """Publishes rendered avatar frames to a v4l2loopback / OBS virtual camera.

    This lets Zoom, Meet, Teams, Discord, OBS, etc. pick up the AI avatar
    as a regular webcam input.
    """

    def __init__(self, settings: Settings, orch: Orchestrator):
        self.settings = settings
        self.orch = orch
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if not self.settings.enable_virtual_camera:
            return
        self._task = asyncio.create_task(self._run(), name="virtualcam")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task

    async def _run(self) -> None:
        import pyvirtualcam

        w = self.settings.virtual_camera_width
        h = self.settings.virtual_camera_height
        fps = self.settings.lipsync_fps
        try:
            cam = pyvirtualcam.Camera(
                width=w,
                height=h,
                fps=fps,
                device=self.settings.virtual_camera_device,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
        except Exception as e:
            logger.error(
                f"virtual camera unavailable ({e}). On linux: "
                "`sudo modprobe v4l2loopback devices=1 card_label=\"OpenAI Video Chat\"`"
            )
            return

        logger.info(f"virtual camera publishing to {cam.device} at {w}x{h}@{fps}fps")
        idle_interval = 1.0 / fps
        pending: list[np.ndarray] = []
        consume_task = asyncio.create_task(self._consume(pending), name="virtualcam.consume")

        try:
            while not self._stop.is_set():
                if pending:
                    frame = pending.pop(0)
                else:
                    frame = self.orch.avatar.idle_frame()
                    if frame is None:
                        frame = np.zeros((h, w, 3), dtype=np.uint8)
                cam.send(frame)
                cam.sleep_until_next_frame()
                await asyncio.sleep(0)  # cooperative yield
        finally:
            consume_task.cancel()
            cam.close()

    async def _consume(self, pending: list[np.ndarray]) -> None:
        async for pair in self.orch.av_stream():
            for vf in pair.video:
                pending.append(vf.image)
