from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from ..config import Settings


class TTSBackend(Protocol):
    sample_rate: int
    speaker_wav: str | None

    def synth(self, text: str) -> np.ndarray: ...


class PiperBackend:
    """Piper TTS — fast, CPU-friendly. Voice ONNX is loaded once and reused."""

    speaker_wav = None

    def __init__(self, settings: Settings):
        from piper import PiperVoice

        self.settings = settings
        model_path = self._resolve_voice(settings.tts_voice, settings.models_dir / "piper")
        logger.info(f"loading piper voice {model_path}")
        self.voice = PiperVoice.load(str(model_path))
        self.sample_rate = self.voice.config.sample_rate
        self._available = self._list_voices(settings.models_dir / "piper")

    @staticmethod
    def _resolve_voice(name: str, voices_dir: Path) -> Path:
        candidate = voices_dir / f"{name}.onnx"
        if not candidate.exists():
            raise FileNotFoundError(
                f"piper voice {candidate} not found. "
                f"Run scripts/setup.sh or scripts/download_models.py to fetch it."
            )
        return candidate

    @staticmethod
    def _list_voices(voices_dir: Path) -> list[str]:
        if not voices_dir.exists():
            return []
        return sorted(p.stem for p in voices_dir.glob("*.onnx"))

    def available_voices(self) -> list[str]:
        return list(self._available)

    def set_voice(self, name: str) -> None:
        from piper import PiperVoice

        path = self._resolve_voice(name, self.settings.models_dir / "piper")
        logger.info(f"switching piper voice to {path}")
        self.voice = PiperVoice.load(str(path))
        self.sample_rate = self.voice.config.sample_rate

    def synth(self, text: str) -> np.ndarray:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if sr != self.sample_rate:
            self.sample_rate = sr
        return pcm


class XTTSBackend:
    """Coqui XTTS-v2 — voice cloning from a 6-30s reference clip."""

    def __init__(self, settings: Settings):
        from TTS.api import TTS as CoquiTTS

        self.settings = settings
        device = settings.device if settings.device != "mps" else "cpu"
        logger.info(f"loading XTTS-v2 on {device}")
        self.tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.sample_rate = 24000
        self.speaker_wav: str | None = settings.xtts_speaker_wav or None

    def available_voices(self) -> list[str]:
        return ["xtts-clone"]

    def set_voice(self, name: str) -> None:
        # xtts uses speaker_wav, not named voices
        pass

    def synth(self, text: str) -> np.ndarray:
        wav = self.tts.tts(
            text=text,
            speaker_wav=self.speaker_wav,
            language="en",
            split_sentences=False,
        )
        return np.asarray(wav, dtype=np.float32)


def build_tts_backend(settings: Settings) -> TTSBackend:
    if settings.tts_backend == "xtts":
        return XTTSBackend(settings)
    return PiperBackend(settings)
