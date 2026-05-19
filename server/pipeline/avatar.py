from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import cv2
import numpy as np
from loguru import logger

from ..config import Settings
from ..utils.audio import resample


@dataclass
class VideoFrame:
    image: np.ndarray   # uint8, HxWx3, BGR
    pts: float          # presentation timestamp in seconds, monotonic per session


@dataclass
class AVPair:
    audio: np.ndarray   # float32 mono at output_sr
    video: list[VideoFrame]
    sample_rate: int


class _BaseLipSync:
    fps: int
    image_size: tuple[int, int]
    output_sr: int = 16000

    def warmup(self, ref_image: np.ndarray) -> None:
        raise NotImplementedError

    def render(self, audio: np.ndarray, sr: int) -> AVPair:
        raise NotImplementedError


class _MuseTalkBackend(_BaseLipSync):
    """MuseTalk: latent-space real-time lip sync (Tencent, MIT licensed).

    The full inference graph is shipped in `models/musetalk/` after running
    `scripts/setup.sh`. We wrap it here in a synchronous `render(audio)`
    call that returns a contiguous video clip aligned to the audio.

    The real MuseTalk pipeline:
      1. Whisper features from audio (chunked).
      2. VAE-encoded reference face crop.
      3. UNet predicts mouth latents conditioned on audio features.
      4. VAE decode → mouth crop → blended back into reference frame.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.fps = settings.lipsync_fps
        self.batch = settings.lipsync_batch
        self.image_size = (settings.virtual_camera_width, settings.virtual_camera_height)
        self._model = None
        self._ref_cache: dict[str, dict] = {}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from .musetalk_runtime import MuseTalkRuntime
        except ImportError as e:
            raise RuntimeError(
                "musetalk runtime not installed. Run scripts/setup.sh to download "
                "the model weights and clone the MuseTalk source."
            ) from e
        logger.info("initializing MuseTalk runtime")
        self._model = MuseTalkRuntime(
            checkpoint_dir=self.settings.models_dir / "musetalk",
            device=self.settings.device,
            fps=self.fps,
            batch=self.batch,
        )

    def warmup(self, ref_image: np.ndarray) -> None:
        self._ensure_model()
        key = f"{ref_image.shape}-{int(ref_image.mean())}"
        if key not in self._ref_cache:
            self._ref_cache[key] = self._model.prepare_reference(ref_image)
        self._ref_key = key

    def render(self, audio: np.ndarray, sr: int) -> AVPair:
        self._ensure_model()
        audio16 = resample(audio, sr, self.output_sr)
        frames = self._model.infer(audio16, self._ref_cache[self._ref_key])
        # Resize to target output and tag PTS
        h, w = self.image_size[1], self.image_size[0]
        out: list[VideoFrame] = []
        dt = 1.0 / self.fps
        for i, f in enumerate(frames):
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            out.append(VideoFrame(image=f, pts=i * dt))
        return AVPair(audio=audio16, video=out, sample_rate=self.output_sr)


class _Wav2LipBackend(_BaseLipSync):
    """Wav2Lip ONNX fallback for CPU / low-VRAM. Lower fidelity but reliable."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.fps = settings.lipsync_fps
        self.image_size = (settings.virtual_camera_width, settings.virtual_camera_height)
        self._session = None
        self._face_box: Optional[tuple[int, int, int, int]] = None
        self._ref: Optional[np.ndarray] = None

    def _ensure(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        path = self.settings.models_dir / "wav2lip" / "wav2lip.onnx"
        if not path.exists():
            raise FileNotFoundError(
                f"wav2lip onnx not found at {path}. Run scripts/setup.sh."
            )
        providers = ["CPUExecutionProvider"]
        if self.settings.device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        logger.info(f"loading wav2lip onnx with {providers[0]}")
        self._session = ort.InferenceSession(str(path), providers=providers)

    def warmup(self, ref_image: np.ndarray) -> None:
        self._ensure()
        import mediapipe as mp

        det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        h, w = ref_image.shape[:2]
        results = det.process(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            raise ValueError("no face detected in reference image")
        bb = results.detections[0].location_data.relative_bounding_box
        x = max(0, int(bb.xmin * w))
        y = max(0, int(bb.ymin * h))
        bw = int(bb.width * w)
        bh = int(bb.height * h)
        self._face_box = (x, y, bw, bh)
        self._ref = ref_image.copy()

    def render(self, audio: np.ndarray, sr: int) -> AVPair:
        self._ensure()
        audio16 = resample(audio, sr, self.output_sr)
        n_frames = max(1, int(np.ceil(len(audio16) / self.output_sr * self.fps)))
        # Mel features for wav2lip
        mel = self._mel(audio16)
        x, y, w, h = self._face_box  # type: ignore[misc]
        face = cv2.resize(self._ref[y:y + h, x:x + w], (96, 96))
        face_b = face.astype(np.float32) / 255.0
        face_b = np.transpose(face_b, (2, 0, 1))[None, :]
        out_frames: list[VideoFrame] = []
        dt = 1.0 / self.fps
        for i in range(n_frames):
            seg = self._mel_slice(mel, i, n_frames)
            seg = seg[None, None, :, :]
            preds = self._session.run(None, {"mel": seg.astype(np.float32), "vid": face_b.astype(np.float32)})[0]
            mouth = (np.transpose(preds[0], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
            mouth = cv2.resize(mouth, (w, h))
            frame = self._ref.copy()
            frame[y:y + h, x:x + w] = mouth
            if frame.shape[:2] != (self.image_size[1], self.image_size[0]):
                frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_LINEAR)
            out_frames.append(VideoFrame(image=frame, pts=i * dt))
        return AVPair(audio=audio16, video=out_frames, sample_rate=self.output_sr)

    @staticmethod
    def _mel(audio: np.ndarray) -> np.ndarray:
        import librosa

        m = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=800, hop_length=200, n_mels=80, fmin=55, fmax=7600)
        m = librosa.power_to_db(m, ref=np.max)
        return ((m + 80.0) / 80.0).astype(np.float32)

    @staticmethod
    def _mel_slice(mel: np.ndarray, i: int, total: int) -> np.ndarray:
        T = mel.shape[1]
        start = int(round(i / total * (T - 16)))
        return mel[:, start:start + 16]


class Avatar:
    """Front-door for the lip-sync pipeline. Async, queue-driven."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.lipsync_backend == "musetalk":
            self.backend: _BaseLipSync = _MuseTalkBackend(settings)
        else:
            self.backend = _Wav2LipBackend(settings)
        self.fps = self.backend.fps
        self._ready = False
        self._idle_frame: Optional[np.ndarray] = None
        self._queue: asyncio.Queue[Optional[AVPair]] = asyncio.Queue(maxsize=4)
        self._cancel = asyncio.Event()

    async def load_persona(self, image_path: Path) -> None:
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"avatar image not found: {image_path}")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.backend.warmup, img)
        self._idle_frame = cv2.resize(img, (self.settings.virtual_camera_width, self.settings.virtual_camera_height))
        self._ready = True
        logger.info(f"persona loaded from {image_path}")

    def idle_frame(self) -> Optional[np.ndarray]:
        return self._idle_frame

    def cancel(self) -> None:
        self._cancel.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def reset(self) -> None:
        self._cancel.clear()

    async def render(self, audio: np.ndarray, sr: int) -> None:
        if not self._ready:
            raise RuntimeError("call load_persona first")
        if self._cancel.is_set():
            return
        loop = asyncio.get_running_loop()
        try:
            pair = await loop.run_in_executor(None, self.backend.render, audio, sr)
        except Exception as e:
            logger.exception(f"avatar render failed: {e}")
            return
        if self._cancel.is_set():
            return
        await self._queue.put(pair)

    async def end(self) -> None:
        await self._queue.put(None)

    async def stream(self) -> AsyncIterator[AVPair]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item
