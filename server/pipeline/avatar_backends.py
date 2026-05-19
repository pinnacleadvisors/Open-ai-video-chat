from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from loguru import logger

from ..config import Settings
from ..utils.audio import resample


@dataclass
class VideoFrame:
    image: np.ndarray   # uint8, HxWx3, BGR
    pts: float


@dataclass
class AVPair:
    audio: np.ndarray   # float32 mono at output_sr
    video: list[VideoFrame]
    sample_rate: int


class AvatarBackend(Protocol):
    fps: int
    image_size: tuple[int, int]
    output_sr: int

    def prepare_reference(self, image_bgr: np.ndarray) -> dict: ...
    def render(self, audio: np.ndarray, sr: int, ref: dict) -> AVPair: ...


class MuseTalkBackend:
    """MuseTalk: latent-space real-time lip sync. Reference cache is per-image."""

    output_sr = 16000

    def __init__(self, settings: Settings):
        self.settings = settings
        self.fps = settings.lipsync_fps
        self.batch = settings.lipsync_batch
        self.image_size = (settings.virtual_camera_width, settings.virtual_camera_height)
        self._model = None

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

    def prepare_reference(self, image_bgr: np.ndarray) -> dict:
        self._ensure_model()
        return self._model.prepare_reference(image_bgr)

    def render(self, audio: np.ndarray, sr: int, ref: dict) -> AVPair:
        self._ensure_model()
        audio16 = resample(audio, sr, self.output_sr)
        frames = self._model.infer(audio16, ref)
        h, w = self.image_size[1], self.image_size[0]
        out: list[VideoFrame] = []
        dt = 1.0 / self.fps
        for i, f in enumerate(frames):
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
            out.append(VideoFrame(image=f, pts=i * dt))
        return AVPair(audio=audio16, video=out, sample_rate=self.output_sr)


class Wav2LipBackend:
    """Wav2Lip ONNX fallback for CPU / low-VRAM."""

    output_sr = 16000

    def __init__(self, settings: Settings):
        self.settings = settings
        self.fps = settings.lipsync_fps
        self.image_size = (settings.virtual_camera_width, settings.virtual_camera_height)
        self._session = None

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

    def prepare_reference(self, image_bgr: np.ndarray) -> dict:
        self._ensure()
        import mediapipe as mp

        det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        h, w = image_bgr.shape[:2]
        results = det.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.detections:
            raise ValueError("no face detected in reference image")
        bb = results.detections[0].location_data.relative_bounding_box
        x = max(0, int(bb.xmin * w))
        y = max(0, int(bb.ymin * h))
        bw = int(bb.width * w)
        bh = int(bb.height * h)
        return {"face_box": (x, y, bw, bh), "frame": image_bgr.copy()}

    def render(self, audio: np.ndarray, sr: int, ref: dict) -> AVPair:
        self._ensure()
        audio16 = resample(audio, sr, self.output_sr)
        n_frames = max(1, int(np.ceil(len(audio16) / self.output_sr * self.fps)))
        mel = self._mel(audio16)
        x, y, w, h = ref["face_box"]
        frame0 = ref["frame"]
        face = cv2.resize(frame0[y:y + h, x:x + w], (96, 96))
        face_b = (face.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, :]
        out_frames: list[VideoFrame] = []
        dt = 1.0 / self.fps
        for i in range(n_frames):
            seg = self._mel_slice(mel, i, n_frames)[None, None, :, :]
            preds = self._session.run(
                None, {"mel": seg.astype(np.float32), "vid": face_b.astype(np.float32)}
            )[0]
            mouth = (np.transpose(preds[0], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
            mouth = cv2.resize(mouth, (w, h))
            frame = frame0.copy()
            frame[y:y + h, x:x + w] = mouth
            if frame.shape[:2] != (self.image_size[1], self.image_size[0]):
                frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_LINEAR)
            out_frames.append(VideoFrame(image=frame, pts=i * dt))
        return AVPair(audio=audio16, video=out_frames, sample_rate=self.output_sr)

    @staticmethod
    def _mel(audio: np.ndarray) -> np.ndarray:
        import librosa

        m = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=800, hop_length=200, n_mels=80, fmin=55, fmax=7600
        )
        m = librosa.power_to_db(m, ref=np.max)
        return ((m + 80.0) / 80.0).astype(np.float32)

    @staticmethod
    def _mel_slice(mel: np.ndarray, i: int, total: int) -> np.ndarray:
        T = mel.shape[1]
        start = int(round(i / total * (T - 16)))
        return mel[:, start:start + 16]


def build_avatar_backend(settings: Settings) -> AvatarBackend:
    if settings.lipsync_backend == "musetalk":
        return MuseTalkBackend(settings)
    return Wav2LipBackend(settings)
