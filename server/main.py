from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from .auth import BearerTokenMiddleware
from .config import Settings, get_settings
from .engines import Engines
from .metrics import metrics
from .personas_store import Persona, PersonaStore
from .pipeline import AvatarSession, LLMSession, Orchestrator, STTSession, TTSSession
from .utils.uploads import save_upload
from .virtual_camera import VirtualCameraPublisher
from .webrtc.session import WebRTCSession


class _AppState:
    settings: Settings
    engines: Engines
    personas: PersonaStore
    virtual_camera: VirtualCameraPublisher
    sessions: dict[str, WebRTCSession]

    def __init__(self) -> None:
        self.sessions = {}


state = _AppState()


def _configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "<level>{level: <8}</level> "
            "<cyan>{extra[session]:<8}</cyan> "
            "<level>{message}</level>"
        ),
        filter=_inject_default_session,
    )


def _inject_default_session(record) -> bool:  # type: ignore[no-untyped-def]
    record["extra"].setdefault("session", "-")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.settings = settings
    _configure_logging(settings)

    settings.media_dir.mkdir(parents=True, exist_ok=True)
    (settings.media_dir / "personas").mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading engines (this may take a minute)")
    state.engines = Engines(settings)
    state.engines.load()

    state.personas = PersonaStore(settings.data_dir / "personas.db")
    state.virtual_camera = VirtualCameraPublisher(settings)

    yield

    logger.info("shutting down")
    for s in list(state.sessions.values()):
        await s.close()
    await state.virtual_camera.stop()
    await state.engines.close()


app = FastAPI(title="open-ai-video-chat", lifespan=lifespan)


@app.middleware("http")
async def _log_requests(request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    return response


def _install_middleware() -> None:
    settings = get_settings()
    app.add_middleware(BearerTokenMiddleware, token=settings.auth_token)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list() or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_install_middleware()


# ---------- health / metrics --------------------------------------------------

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


@app.get("/api/ready")
async def ready() -> JSONResponse:
    """Readiness: engines loaded AND at least one persona reachable."""
    if not hasattr(state, "engines") or state.engines.whisper is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ready"})


@app.get("/api/metrics")
async def metrics_endpoint() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


# ---------- personas ----------------------------------------------------------

class PersonaCreate(BaseModel):
    name: str
    voice: str
    speaker_wav: str | None = None
    system_prompt: str | None = None


@app.get("/api/personas")
async def list_personas() -> list[dict]:
    return [p.__dict__ for p in state.personas.list()]


@app.post("/api/personas")
async def create_persona(
    file: UploadFile,
    name: str,
    voice: str,
    speaker_wav: str | None = None,
    system_prompt: str | None = None,
) -> dict:
    persona_id = uuid.uuid4().hex
    image_path = save_upload(
        file,
        state.settings.media_dir / "personas",
        basename=persona_id,
        kind="image",
        max_bytes=state.settings.max_image_upload_mb * (1 << 20),
    )
    persona = Persona(
        id=persona_id,
        name=name,
        image_path=str(image_path.relative_to(state.settings.root_dir)),
        voice=voice,
        speaker_wav=speaker_wav,
        system_prompt=system_prompt,
    )
    state.personas.upsert(persona)
    return persona.__dict__


@app.delete("/api/personas/{persona_id}")
async def delete_persona(persona_id: str) -> dict:
    return {"deleted": state.personas.delete(persona_id)}


@app.post("/api/persona/voice")
async def upload_voice_sample(file: UploadFile) -> dict:
    dst = save_upload(
        file,
        state.settings.media_dir / "personas",
        basename=f"voice-{uuid.uuid4().hex}",
        kind="audio",
        max_bytes=state.settings.max_audio_upload_mb * (1 << 20),
    )
    return {"path": str(dst.relative_to(state.settings.root_dir))}


# ---------- voices ------------------------------------------------------------

@app.get("/api/voices")
async def list_voices() -> dict:
    backend = state.engines.tts_backend
    voices = backend.available_voices() if hasattr(backend, "available_voices") else []
    return {"backend": state.settings.tts_backend, "voices": voices}


# ---------- webrtc ------------------------------------------------------------

class OfferBody(BaseModel):
    sdp: str
    type: str
    persona_id: str | None = None


def _make_orchestrator(persona: Persona | None) -> Orchestrator:
    settings = state.settings
    engines = state.engines
    session_id = uuid.uuid4().hex[:8]
    stt = STTSession(settings, engines.whisper, engines.vad, session_id)
    llm = LLMSession(settings, engines.http, session_id)
    tts = TTSSession(settings, engines.tts_backend, session_id)
    avatar = AvatarSession(settings, engines.avatar_backend, session_id)
    orch = Orchestrator(settings, stt, llm, tts, avatar, session_id)

    if persona:
        if persona.system_prompt:
            llm.update_system_prompt(persona.system_prompt)
        if persona.voice and hasattr(engines.tts_backend, "set_voice"):
            try:
                engines.tts_backend.set_voice(persona.voice)
            except Exception as e:
                logger.bind(session=session_id).warning(f"voice switch failed: {e}")
        if persona.speaker_wav and hasattr(engines.tts_backend, "speaker_wav"):
            engines.tts_backend.speaker_wav = persona.speaker_wav
    return orch


async def _load_persona_image(orch: Orchestrator, persona: Persona | None) -> None:
    image_path: Path | None = None
    if persona:
        image_path = state.settings.root_dir / persona.image_path
    elif state.settings.avatar_path().exists():
        image_path = state.settings.avatar_path()
    if image_path and image_path.exists():
        await orch.avatar.load_persona(image_path)


@app.post("/api/webrtc/offer")
async def webrtc_offer(body: OfferBody) -> JSONResponse:
    if len(state.sessions) >= state.settings.max_sessions:
        metrics.webrtc_negotiation_failures_total.inc()
        raise HTTPException(503, "session limit reached")
    metrics.webrtc_negotiations_total.inc()

    persona = state.personas.get(body.persona_id) if body.persona_id else None
    orch = _make_orchestrator(persona)
    await _load_persona_image(orch, persona)
    await orch.start()

    session = WebRTCSession(state.settings, orch)
    state.sessions[session.id] = session

    if not state.virtual_camera._task and state.settings.enable_virtual_camera:
        state.virtual_camera.attach(orch)

    try:
        answer = await session.offer(body.sdp, body.type)
    except Exception:
        metrics.webrtc_negotiation_failures_total.inc()
        await session.close()
        state.sessions.pop(session.id, None)
        raise
    return JSONResponse({"id": session.id, **answer})


@app.delete("/api/webrtc/{session_id}")
async def webrtc_close(session_id: str) -> dict:
    session = state.sessions.pop(session_id, None)
    if session:
        await session.close()
    return {"ok": True}


# ---------- transcripts ws ----------------------------------------------------

@app.websocket("/ws/transcripts/{session_id}")
async def transcripts_ws(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    session = state.sessions.get(session_id)
    if not session:
        await ws.send_json({"error": "session not found"})
        await ws.close()
        return

    sub = session.orch.transcripts.subscribe()
    try:
        async for ev in sub:
            await ws.send_json({"role": ev.role, "text": ev.text, "final": ev.final})
    except WebSocketDisconnect:
        return
    except Exception as e:
        logger.exception(f"transcripts ws error: {e}")
    finally:
        await sub.aclose()
        try:
            await ws.close()
        except Exception:
            pass


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
