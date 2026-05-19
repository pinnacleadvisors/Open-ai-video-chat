from __future__ import annotations

import asyncio
import fractions
import time
import uuid
from typing import Optional

import av
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import MediaStreamError
from loguru import logger

from ..config import Settings
from ..pipeline import Orchestrator
from ..utils.audio import to_mono_f32, resample, s16_from_f32


class _AvatarVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, settings: Settings, orch: Orchestrator):
        super().__init__()
        self.settings = settings
        self.orch = orch
        self._fps = orch.avatar.fps
        self._frame_interval = 1.0 / self._fps
        self._t0 = time.monotonic()
        self._pts = 0
        self._time_base = fractions.Fraction(1, 90000)
        self._pending: list[np.ndarray] = []
        self._task = asyncio.create_task(self._pump(), name="avatar.video.pump")

    async def _pump(self) -> None:
        async for pair in self.orch.av_stream():
            for f in pair.video:
                self._pending.append(f.image)

    def _next_image(self) -> np.ndarray:
        if self._pending:
            return self._pending.pop(0)
        idle = self.orch.avatar.idle_frame()
        if idle is None:
            return np.zeros((self.settings.virtual_camera_height, self.settings.virtual_camera_width, 3), dtype=np.uint8)
        return idle

    async def recv(self) -> av.VideoFrame:
        target = self._t0 + self._pts / self._fps * 1.0
        now = time.monotonic()
        if now < target:
            await asyncio.sleep(target - now)

        img = self._next_image()
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = int(self._pts * 90000 / self._fps)
        frame.time_base = self._time_base
        self._pts += 1
        return frame


class _AvatarAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, orch: Orchestrator):
        super().__init__()
        self.orch = orch
        self._sample_rate = 48000
        self._samples_per_frame = 960   # 20ms at 48k
        self._time_base = fractions.Fraction(1, self._sample_rate)
        self._buffer = np.zeros(0, dtype=np.int16)
        self._pts = 0
        self._task = asyncio.create_task(self._pump(), name="avatar.audio.pump")

    async def _pump(self) -> None:
        async for pair in self.orch.av_stream():
            pcm48 = resample(pair.audio, pair.sample_rate, self._sample_rate)
            self._buffer = np.concatenate([self._buffer, s16_from_f32(pcm48)])

    async def recv(self) -> av.AudioFrame:
        if self._buffer.size < self._samples_per_frame:
            pad = np.zeros(self._samples_per_frame - self._buffer.size, dtype=np.int16)
            chunk = np.concatenate([self._buffer, pad])
            self._buffer = np.zeros(0, dtype=np.int16)
            # pace silence at real-time
            await asyncio.sleep(self._samples_per_frame / self._sample_rate)
        else:
            chunk = self._buffer[: self._samples_per_frame]
            self._buffer = self._buffer[self._samples_per_frame :]

        frame = av.AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.sample_rate = self._sample_rate
        frame.planes[0].update(chunk.tobytes())
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += self._samples_per_frame
        return frame


class WebRTCSession:
    """One peer connection per browser tab."""

    def __init__(self, settings: Settings, orch: Orchestrator):
        self.id = str(uuid.uuid4())
        self.settings = settings
        self.orch = orch
        cfg = RTCConfiguration(iceServers=[RTCIceServer(urls=s["urls"]) for s in settings.ice_server_list()])
        self.pc = RTCPeerConnection(configuration=cfg)
        self._inbound: Optional[asyncio.Task] = None

        self.pc.addTrack(_AvatarVideoTrack(settings, orch))
        self.pc.addTrack(_AvatarAudioTrack(orch))

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":
                logger.info(f"session {self.id} mic track received")
                self._inbound = asyncio.create_task(self._consume_audio(track), name="webrtc.consume")

        @self.pc.on("connectionstatechange")
        async def on_state() -> None:
            logger.info(f"session {self.id} connection state {self.pc.connectionState}")
            if self.pc.connectionState in ("failed", "closed", "disconnected"):
                await self.close()

    async def offer(self, sdp: str, type_: str) -> dict:
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}

    async def _consume_audio(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame: av.AudioFrame = await track.recv()
                arr = frame.to_ndarray()  # shape (channels, samples) for s16
                if arr.ndim == 2:
                    arr = arr[0]
                pcm = to_mono_f32(arr)
                if frame.sample_rate != 16000:
                    pcm = resample(pcm, frame.sample_rate, 16000)
                self.orch.push_audio(pcm)
        except MediaStreamError:
            logger.info(f"session {self.id} mic ended")

    async def close(self) -> None:
        if self._inbound:
            self._inbound.cancel()
        await self.pc.close()
