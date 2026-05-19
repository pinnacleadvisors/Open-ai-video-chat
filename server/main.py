from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from .config import Settings, get_settings
from .pipeline import STT, LLM, TTS, Avatar, Orchestrator
from .virtual_camera import VirtualCameraPublisher
from .webrtc.session import WebRTCSession


class _AppState:
    settings: Settings
    stt: STT
    llm: LLM
    tts: TTS
    avatar: Avatar
    orch: Orchestrator
    virtual_camera: VirtualCameraPublisher
    sessions: dict[str, WebRTCSession]

    def __init__(self) -> None:
        self.sessions = {}


state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.settings = settings

    settings.media_dir.mkdir(parents=True, exist_ok=True)
    (settings.media_dir / "personas").mkdir(parents=True, exist_ok=True)

    logger.info("initializing pipeline")
    state.stt = STT(settings)
    state.llm = LLM(settings)
    state.tts = TTS(settings)
    state.avatar = Avatar(settings)

    avatar_path = settings.avatar_path()
    if avatar_path.exists():
        await state.avatar.load_persona(avatar_path)
    else:
        logger.warning(f"no default avatar at {avatar_path} — upload one via /api/persona")

    state.orch = Orchestrator(settings, state.stt, state.llm, state.tts, state.avatar)
    await state.orch.start()

    state.virtual_camera = VirtualCameraPublisher(settings, state.orch)
    await state.virtual_camera.start()

    yield

    logger.info("shutting down")
    for s in list(state.sessions.values()):
        await s.close()
    await state.virtual_camera.stop()
    await state.orch.stop()
    await state.llm.close()


app = FastAPI(title="open-ai-video-chat", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    s = state.settings
    return {
        "status": "ok",
        "device": s.device,
        "llm": {"backend": s.llm_backend, "model": s.llm_model},
        "tts": {"backend": s.tts_backend, "voice": s.tts_voice},
        "lipsync": {"backend": s.lipsync_backend, "fps": s.lipsync_fps},
        "stt": {"model": s.stt_model},
        "sessions": len(state.sessions),
    }


class PersonaConfig(BaseModel):
    system_prompt: Optional[str] = None
    voice: Optional[str] = None
    speaker_wav: Optional[str] = None


@app.post("/api/persona/image")
async def upload_persona_image(file: UploadFile) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "file must be an image")
    dst = state.settings.media_dir / "personas" / f"persona{Path(file.filename or 'persona.png').suffix}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    await state.avatar.load_persona(dst)
    return {"path": str(dst.relative_to(state.settings.root_dir))}


@app.post("/api/persona/voice")
async def upload_voice_sample(file: UploadFile) -> dict:
    """Upload a 6–30s clean voice sample for XTTS cloning."""
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(400, "file must be an audio clip")
    dst = state.settings.media_dir / "personas" / f"voice{Path(file.filename or 'voice.wav').suffix}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    if hasattr(state.tts.backend, "speaker_wav"):
        state.tts.backend.speaker_wav = str(dst)
    return {"path": str(dst.relative_to(state.settings.root_dir))}


@app.post("/api/persona/config")
async def update_persona(cfg: PersonaConfig) -> dict:
    if cfg.system_prompt:
        state.llm.update_system_prompt(cfg.system_prompt)
    if cfg.speaker_wav and hasattr(state.tts.backend, "speaker_wav"):
        state.tts.backend.speaker_wav = cfg.speaker_wav
    return {"ok": True}


class OfferBody(BaseModel):
    sdp: str
    type: str


@app.post("/api/webrtc/offer")
async def webrtc_offer(body: OfferBody) -> JSONResponse:
    session = WebRTCSession(state.settings, state.orch)
    state.sessions[session.id] = session
    answer = await session.offer(body.sdp, body.type)
    return JSONResponse({"id": session.id, **answer})


@app.delete("/api/webrtc/{session_id}")
async def webrtc_close(session_id: str) -> dict:
    session = state.sessions.pop(session_id, None)
    if session:
        await session.close()
    return {"ok": True}


@app.websocket("/ws/transcripts")
async def transcripts_ws(ws: WebSocket) -> None:
    """Live transcript stream — useful for the UI's chat panel."""
    await ws.accept()
    try:
        async for ev in state.orch.transcripts():
            await ws.send_json({"role": ev.role, "text": ev.text, "final": ev.final})
    except WebSocketDisconnect:
        return
    except Exception as e:
        logger.exception(f"transcripts ws error: {e}")
        await ws.close()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
