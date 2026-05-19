from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Pipeline / WebRTC / virtual-cam imports pull in cv2/torch/aiortc; do them
# lazily inside the handlers so this module loads in lightweight envs (tests).
from typing import TYPE_CHECKING

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from .auth import BearerTokenMiddleware
from .config import Settings, get_settings
from .engines import Engines
from .health_probes import run_probes
from .lifecycle import graceful_drain
from .metrics import metrics
from .middleware import RequestIdMiddleware, SecurityHeadersMiddleware
from .personas_store import Persona, PersonaStore
from .rate_limit import TokenBucket
from .rate_limit import enforce as enforce_rate
from .utils.uploads import save_upload

if TYPE_CHECKING:
    from .pipeline import Orchestrator
    from .virtual_camera import VirtualCameraPublisher
    from .webrtc.session import WebRTCSession


class _AppState:
    settings: Settings
    engines: Engines
    personas: PersonaStore
    virtual_camera: VirtualCameraPublisher
    sessions: dict[str, WebRTCSession]
    offer_bucket: TokenBucket
    upload_bucket: TokenBucket

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
            "<dim>{extra[request_id]:<12}</dim> "
            "<level>{message}</level>"
        ),
        filter=_inject_defaults,
    )


def _inject_defaults(record) -> bool:  # type: ignore[no-untyped-def]
    record["extra"].setdefault("session", "-")
    record["extra"].setdefault("request_id", "-")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.settings = settings
    _configure_logging(settings)

    settings.media_dir.mkdir(parents=True, exist_ok=True)
    (settings.media_dir / "personas").mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    from .virtual_camera import VirtualCameraPublisher

    state.engines = Engines(settings)
    state.personas = PersonaStore(settings.data_dir / "personas.db")
    state.virtual_camera = VirtualCameraPublisher(settings)
    state.offer_bucket = TokenBucket(
        rate=settings.rate_limit_offer_per_min / 60.0,
        capacity=settings.rate_limit_offer_per_min,
    )
    state.upload_bucket = TokenBucket(
        rate=settings.rate_limit_upload_per_min / 60.0,
        capacity=settings.rate_limit_upload_per_min,
    )

    if os.environ.get("OAVC_SKIP_ENGINES") != "1":
        logger.info("loading engines (this may take a minute)")
        state.engines.load()

    yield

    logger.info("shutdown beginning, draining sessions")
    await graceful_drain(_close_all_sessions, timeout=settings.shutdown_drain_s)
    await state.virtual_camera.stop()
    await state.engines.close()
    logger.info("shutdown complete")


async def _close_all_sessions() -> None:
    sessions = list(state.sessions.values())
    state.sessions.clear()
    if not sessions:
        return
    await asyncio.gather(*(s.close() for s in sessions), return_exceptions=True)


app = FastAPI(title="open-ai-video-chat", lifespan=lifespan)


def _install_middleware() -> None:
    settings = get_settings()
    # Middleware runs in *reverse* registration order. We want, on the way IN:
    #   RequestId -> Auth -> CORS -> SecurityHeaders -> app
    # So register them inner-first (outer-last):
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.enable_hsts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list() or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BearerTokenMiddleware, token=lambda: get_settings().auth_token)
    app.add_middleware(RequestIdMiddleware)


_install_middleware()


def _rl_offer(request: Request) -> None:
    enforce_rate(state.offer_bucket, request)


def _rl_upload(request: Request) -> None:
    enforce_rate(state.upload_bucket, request)


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
    engines_ready = hasattr(state, "engines") and state.engines.whisper is not None
    report = await run_probes(state.settings, state.engines.http, engines_ready=engines_ready)
    code = 200 if report.ready else 503
    return JSONResponse(report.to_dict(), status_code=code)


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


@app.post("/api/personas", dependencies=[Depends(_rl_upload)])
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
        image_path=str(image_path),  # absolute on disk
        voice=voice,
        speaker_wav=speaker_wav,
        system_prompt=system_prompt,
    )
    state.personas.upsert(persona)
    return persona.__dict__


@app.delete("/api/personas/{persona_id}")
async def delete_persona(persona_id: str) -> dict:
    return {"deleted": state.personas.delete(persona_id)}


@app.post("/api/persona/voice", dependencies=[Depends(_rl_upload)])
async def upload_voice_sample(file: UploadFile) -> dict:
    dst = save_upload(
        file,
        state.settings.media_dir / "personas",
        basename=f"voice-{uuid.uuid4().hex}",
        kind="audio",
        max_bytes=state.settings.max_audio_upload_mb * (1 << 20),
    )
    return {"path": str(dst)}


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
    from .pipeline import AvatarSession, LLMSession, Orchestrator, STTSession, TTSSession

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
        candidate = Path(persona.image_path)
        image_path = candidate if candidate.is_absolute() else state.settings.root_dir / candidate
    elif state.settings.avatar_path().exists():
        image_path = state.settings.avatar_path()
    if image_path and image_path.exists():
        await orch.avatar.load_persona(image_path)


@app.post("/api/webrtc/offer", dependencies=[Depends(_rl_offer)])
async def webrtc_offer(body: OfferBody) -> JSONResponse:
    if len(state.sessions) >= state.settings.max_sessions:
        metrics.webrtc_negotiation_failures_total.inc()
        raise HTTPException(503, "session limit reached")
    metrics.webrtc_negotiations_total.inc()

    persona = state.personas.get(body.persona_id) if body.persona_id else None
    try:
        orch = _make_orchestrator(persona)
        await _load_persona_image(orch, persona)
        await orch.start()
    except Exception as e:
        metrics.webrtc_negotiation_failures_total.inc()
        logger.exception(f"orchestrator init failed: {e}")
        raise HTTPException(500, f"orchestrator init failed: {e}")

    from .webrtc.session import WebRTCSession  # noqa: PLC0415 — lazy import for test env

    session = WebRTCSession(state.settings, orch)
    state.sessions[session.id] = session

    if state.virtual_camera._task is None and state.settings.enable_virtual_camera:
        state.virtual_camera.attach(orch)

    try:
        answer = await session.offer(body.sdp, body.type)
    except Exception as e:
        metrics.webrtc_negotiation_failures_total.inc()
        await session.close()
        state.sessions.pop(session.id, None)
        raise HTTPException(400, f"sdp negotiation failed: {e}")
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
        timeout_graceful_shutdown=int(settings.shutdown_drain_s + 5),
    )


if __name__ == "__main__":
    main()
