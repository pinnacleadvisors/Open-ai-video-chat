# open-ai-video-chat

Self-hosted, fully open-source real-time AI avatar for video calls.

A from-scratch replacement for HeyGen Interactive Avatars, Akool
Streaming Avatar, LiveAvatar, and xpression camera built entirely on
permissively-licensed open-source models. The avatar listens, thinks,
and speaks back over a WebRTC video stream with sub-second
turn-taking and barge-in, and can be exposed as a virtual webcam so it
appears as a participant in Zoom, Google Meet, Microsoft Teams, OBS,
Discord, etc.

## What it does

1. **Personas** — Upload a portrait photo or short clip. Build a library
   of personas with per-persona voice, system prompt, and (optionally) a
   cloned voice. The persona library is persisted in SQLite.
2. **Voices** — Pick from any Piper voice you have on disk, or clone
   your own from a 6-30s sample with XTTS-v2.
3. **Brain** — Plug into a local LLM through Ollama (Llama 3.x,
   Qwen2.5, Mistral, …) or any OpenAI-compatible endpoint. The
   conversation runs through a streaming orchestrator that
   interrupts the avatar mid-sentence when the user starts talking.
4. **Camera** — The rendered avatar video is published to:
   - the browser via WebRTC (built-in call UI), and/or
   - a v4l2loopback / OBS virtual camera so any conferencing app can
     select it as a webcam.

## Pipeline

```
  Mic ──► Browser ──WebRTC──► aiortc ──► Silero VAD ──► faster-whisper
                                                            │
                                                            ▼
                                                       Ollama / LLM
                                                            │ (token stream)
                                                            ▼
                                              Piper  /  XTTS-v2  (TTS)
                                                            │ (audio chunks)
                                                            ▼
                                          MuseTalk real-time lip-sync
                                                            │ (video frames)
                       ┌────────────────────────────────────┤
                       ▼                                    ▼
              Browser (WebRTC)                  Virtual camera (v4l2loopback)
                                                            │
                                                            ▼
                                                Zoom / Meet / Teams / OBS
```

Every component is swappable via `.env`. Defaults are tuned for a
single consumer GPU (≥ 8 GB VRAM); CPU-only mode swaps MuseTalk for
Wav2Lip-onnx and XTTS for Piper.

## Architecture

The backend hoists all heavy model loading into a singleton `Engines`
object at process start, and creates a fresh `Session` (own `STTSession`,
`LLMSession`, `TTSSession`, `AvatarSession`, `Orchestrator`) per
WebRTC peer connection. Avatar frames and transcripts are published via
a fan-out `Broadcaster` so the video track, audio track, virtual camera,
and transcript WebSocket each get their own bounded queue — slow
consumers drop oldest frames rather than stalling the pipeline.

```
Engines (singleton, ~6 GB VRAM)
  ├─ faster-whisper
  ├─ silero VAD
  ├─ TTS backend (Piper voice cache | XTTS-v2)
  ├─ Avatar backend (MuseTalk runtime | Wav2Lip ONNX)
  └─ httpx.AsyncClient (shared LLM connection pool)

Session (per WebRTC peer)
  ├─ STTSession   (VAD state machine + buffers + utterance queue)
  ├─ LLMSession   (ChatState + cancel event)
  ├─ TTSSession   (Phraser + audio queue + cancel event)
  ├─ AvatarSession(reference cache + Broadcaster[AVPair])
  └─ Orchestrator (turn-taking, barge-in, transcript Broadcaster)
```

## Open-source components

| Stage           | Default                          | Alt                          |
|-----------------|----------------------------------|------------------------------|
| WebRTC          | aiortc                           | LiveKit (self-hosted)        |
| VAD             | Silero VAD                       | WebRTC VAD                   |
| STT             | faster-whisper (large-v3-turbo)  | whisper.cpp                  |
| LLM             | Ollama (Llama 3.1 8B)            | Any OpenAI-compatible API    |
| TTS             | Piper (CPU) / XTTS-v2 (clone)    | StyleTTS 2, Kokoro           |
| Lip-sync        | MuseTalk (real-time)             | Wav2Lip, LivePortrait        |
| Virtual camera  | pyvirtualcam + v4l2loopback      | OBS Virtual Camera           |
| UI              | Next.js + React                  | -                            |

## Quick start

```bash
git clone <this-repo>
cd open-ai-video-chat
cp .env.example .env

# install Python + JS deps and fetch models (~6 GB)
./scripts/setup.sh

# start ollama and pull a model
ollama serve &
ollama pull llama3.1:8b

# run the stack
./scripts/start.sh
```

Open `http://localhost:3000`, create a persona, hit **Start Call**.

To use the avatar in Zoom/Meet/Teams:

```bash
# linux
sudo modprobe v4l2loopback devices=1 card_label="OpenAI Video Chat"
./scripts/start.sh --virtual-camera
```

Select "OpenAI Video Chat" as your webcam in the conferencing app.

## API

| Endpoint | Description |
|---|---|
| `GET  /api/health` | Liveness + configured backends + active session count |
| `GET  /api/ready` | 200 once engines are loaded, 503 otherwise |
| `GET  /api/metrics` | Prometheus text format |
| `GET  /api/voices` | List available Piper voices on disk |
| `GET  /api/personas` | List personas |
| `POST /api/personas` | Create persona (multipart: `file=<image>`, query: `name`, `voice`, `speaker_wav?`, `system_prompt?`) |
| `DELETE /api/personas/{id}` | Delete persona |
| `POST /api/persona/voice` | Upload a voice sample for cloning |
| `POST /api/webrtc/offer` | Negotiate a WebRTC session. Body: `{sdp, type, persona_id?}` |
| `DELETE /api/webrtc/{session_id}` | Close a session |
| `WS   /ws/transcripts/{session_id}` | Live transcript stream |

If `AUTH_TOKEN` is set, all non-public endpoints require
`Authorization: Bearer <token>` (or `?token=<token>` on the websocket).

## Configuration

All knobs live in `.env`. Notable groups:

- **Auth / CORS**: `AUTH_TOKEN`, `CORS_ORIGINS`
- **Limits**: `MAX_IMAGE_UPLOAD_MB`, `MAX_AUDIO_UPLOAD_MB`, `MAX_SESSIONS`
- **WebRTC**: `STUN_URL`, `TURN_URL`, `TURN_USERNAME`, `TURN_CREDENTIAL`
- **LLM**: `LLM_BACKEND` (`ollama` / `openai`), `LLM_MODEL`, `LLM_BASE_URL`
- **TTS**: `TTS_BACKEND` (`piper` / `xtts`), `TTS_VOICE`
- **Lip-sync**: `LIPSYNC_BACKEND` (`musetalk` / `wav2lip`)
- **STT**: `STT_MODEL`, `VAD_THRESHOLD`, `VAD_SILENCE_MS`
- **Behavior**: `BARGE_IN`, `RESPONSE_DELAY_MS`

## Production deployment

- Set `AUTH_TOKEN` to a strong random value and `CORS_ORIGINS` to your
  production domain. Public endpoints (`/api/health`, `/api/ready`,
  `/api/metrics`) bypass auth.
- Configure a TURN server (coturn, Twilio NTS, Cloudflare Realtime) via
  `TURN_URL` etc. Production WebRTC will fail without one in many NAT
  configurations.
- Scrape `/api/metrics` from Prometheus. Useful series:
  `oavc_sessions_active`, `oavc_utterances_total`,
  `oavc_barge_ins_total`, `oavc_llm_ttft_seconds`,
  `oavc_avatar_render_seconds`.
- Front the server with nginx/Caddy for TLS termination.
- For multi-replica deploys, scale Ollama horizontally and run one
  avatar process per GPU; each process holds the model weights in VRAM.

## Development

```bash
# Python lint + tests
python -m ruff check server tests scripts
python -m pytest tests/ -q

# Frontend typecheck + build
( cd web && npx tsc --noEmit && npm run build )
```

CI runs all four on every PR (`.github/workflows/ci.yml`).

## License

MIT. All bundled model checkpoints retain their own licenses — see
`models/LICENSES.md` after running `scripts/setup.sh`.
