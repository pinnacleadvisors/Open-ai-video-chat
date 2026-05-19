from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    device: Literal["cuda", "cpu", "mps"] = "cuda"

    # WebRTC
    ice_servers: str = "stun:stun.l.google.com:19302"

    # STT
    stt_model: str = "large-v3-turbo"
    stt_compute_type: str = "float16"
    stt_language: str = "auto"
    vad_threshold: float = 0.55
    vad_silence_ms: int = 450

    # LLM
    llm_backend: Literal["ollama", "openai"] = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_system_prompt: str = (
        "You are a friendly, concise video-call assistant. "
        "Speak naturally, like a person. Keep replies under 3 sentences "
        "unless asked for detail."
    )
    llm_temperature: float = 0.7
    llm_max_tokens: int = 256

    # TTS
    tts_backend: Literal["piper", "xtts"] = "piper"
    tts_voice: str = "en_US-amy-medium"
    tts_sample_rate: int = 22050
    xtts_speaker_wav: str = ""

    # Lip-sync
    lipsync_backend: Literal["musetalk", "wav2lip"] = "musetalk"
    lipsync_fps: int = 25
    lipsync_batch: int = 4
    avatar_image: str = "media/personas/default.png"

    # Behavior
    barge_in: bool = True
    response_delay_ms: int = 120

    # Virtual camera
    enable_virtual_camera: bool = False
    virtual_camera_device: str = "/dev/video10"
    virtual_camera_width: int = 512
    virtual_camera_height: int = 512

    # Paths
    root_dir: Path = Field(default=ROOT)
    models_dir: Path = Field(default=ROOT / "models" / "checkpoints")
    media_dir: Path = Field(default=ROOT / "media")

    def ice_server_list(self) -> list[dict]:
        return [{"urls": url.strip()} for url in self.ice_servers.split(",") if url.strip()]

    def avatar_path(self) -> Path:
        p = Path(self.avatar_image)
        return p if p.is_absolute() else self.root_dir / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
