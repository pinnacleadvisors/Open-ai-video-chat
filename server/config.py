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

    # Auth / CORS
    auth_token: str = ""
    cors_origins: str = "http://localhost:3000"

    # Limits
    max_image_upload_mb: int = 16
    max_audio_upload_mb: int = 32
    max_sessions: int = 16

    # WebRTC
    stun_url: str = "stun:stun.l.google.com:19302"
    turn_url: str = ""
    turn_username: str = ""
    turn_credential: str = ""

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
    llm_max_retries: int = 1
    llm_retry_backoff_s: float = 0.4

    # Rate limits (requests per minute per client IP)
    rate_limit_offer_per_min: int = 30
    rate_limit_upload_per_min: int = 10

    # Graceful shutdown
    shutdown_drain_s: float = 15.0

    # HSTS toggle (enable when serving over HTTPS)
    enable_hsts: bool = False

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
    data_dir: Path = Field(default=ROOT / "data")

    def ice_server_list(self) -> list[dict]:
        servers: list[dict] = []
        if self.stun_url:
            servers.append({"urls": self.stun_url})
        if self.turn_url:
            entry: dict = {"urls": self.turn_url}
            if self.turn_username:
                entry["username"] = self.turn_username
            if self.turn_credential:
                entry["credential"] = self.turn_credential
            servers.append(entry)
        return servers

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def avatar_path(self) -> Path:
        p = Path(self.avatar_image)
        return p if p.is_absolute() else self.root_dir / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
