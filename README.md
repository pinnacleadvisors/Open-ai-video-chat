# open-ai-video-chat

Self-hosted, fully open-source real-time AI avatar for video calls.

This is a from-scratch replacement for HeyGen Interactive Avatars, Akool
Streaming Avatar, LiveAvatar, and xpression camera built entirely on
permissively-licensed open-source models. The avatar listens, thinks,
and speaks back over a WebRTC video stream with sub-second
turn-taking, and can be exposed as a virtual webcam so it appears as
a participant in Zoom, Google Meet, Microsoft Teams, OBS, Discord,
etc.

## What it does

1. **Persona** — Upload a portrait photo or short clip. We map that
   into a driving identity used by the lip-sync renderer.
2. **Voice** — Pick a built-in voice or clone your own from a 6-30s
   sample (XTTS-v2). Built-in voices use Piper for low-latency CPU
   inference.
3. **Brain** — Plug into a local LLM through Ollama (Llama 3.x,
   Qwen2.5, Mistral, …) or any OpenAI-compatible endpoint. The
   conversation runs through a streaming orchestrator that
   interrupts the avatar mid-sentence when the user starts talking
   (barge-in).
4. **Camera** — The rendered avatar video is published to:
   - the browser via WebRTC (built-in call UI), and/or
   - a v4l2loopback / OBS virtual camera so any conferencing app can
     select it as a webcam.

## The pipeline

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

Every component is swappable. Defaults are tuned for a single
consumer GPU (≥ 8 GB VRAM); CPU-only mode swaps MuseTalk for
Wav2Lip-onnx and XTTS for Piper.

## Open-source components used

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

Open `http://localhost:3000`, drop in a portrait, pick a voice, and
hit **Start Call**.

To use the avatar in Zoom/Meet/Teams:

```bash
# linux
sudo modprobe v4l2loopback devices=1 card_label="OpenAI Video Chat"
./scripts/start.sh --virtual-camera
```

Select "OpenAI Video Chat" as your webcam in the conferencing app.

## Configuration

All knobs live in `.env`. The most useful ones:

- `LLM_BACKEND` — `ollama` (default) or `openai`
- `LLM_MODEL` — e.g. `llama3.1:8b`, `qwen2.5:7b`, `gpt-4o-mini`
- `TTS_BACKEND` — `piper` (fast CPU) or `xtts` (voice cloning)
- `LIPSYNC_BACKEND` — `musetalk` (GPU) or `wav2lip` (CPU/GPU)
- `STT_MODEL` — `large-v3-turbo`, `medium`, `small.en`, …
- `BARGE_IN` — `true` to interrupt the avatar when the user speaks

## License

MIT. All bundled model checkpoints retain their own licenses — see
`models/LICENSES.md` after running `scripts/setup.sh`.
