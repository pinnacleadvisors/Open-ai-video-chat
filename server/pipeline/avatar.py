from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from ..config import Settings
from ..utils.fanout import Broadcaster
from .avatar_backends import AvatarBackend, AVPair


class AvatarSession:
    """Per-conversation avatar render state. Fan-outs AVPair to many consumers."""

    def __init__(self, settings: Settings, backend: AvatarBackend, session_id: str):
        self.settings = settings
        self.backend = backend
        self.session_id = session_id
        self.fps = backend.fps
        self._ref: dict | None = None
        self._idle_frame: np.ndarray | None = None
        self.frames: Broadcaster[AVPair] = Broadcaster(maxsize=4)
        self._cancel = asyncio.Event()

    async def load_persona(self, image_path: Path) -> None:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"avatar image not found: {image_path}")
        loop = asyncio.get_running_loop()
        self._ref = await loop.run_in_executor(None, self.backend.prepare_reference, img)
        self._idle_frame = cv2.resize(
            img, (self.settings.virtual_camera_width, self.settings.virtual_camera_height)
        )
        logger.bind(session=self.session_id).info(f"persona loaded from {image_path}")

    @property
    def ready(self) -> bool:
        return self._ref is not None

    def idle_frame(self) -> np.ndarray | None:
        return self._idle_frame

    def cancel(self) -> None:
        self._cancel.set()

    def reset(self) -> None:
        self._cancel.clear()

    async def render(self, audio: np.ndarray, sr: int) -> None:
        if not self.ready:
            raise RuntimeError("call load_persona first")
        if self._cancel.is_set():
            return
        loop = asyncio.get_running_loop()
        try:
            pair = await loop.run_in_executor(None, self.backend.render, audio, sr, self._ref)
        except Exception as e:
            logger.bind(session=self.session_id).exception(f"avatar render failed: {e}")
            return
        if self._cancel.is_set():
            return
        await self.frames.publish(pair)

    async def close(self) -> None:
        await self.frames.close()
