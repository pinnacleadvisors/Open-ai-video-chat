import os
import sys
from pathlib import Path

# Make the repo root importable as `server.*`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Prevent server.config from reading an arbitrary .env on a contributor's machine
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("LLM_BACKEND", "ollama")
os.environ.setdefault("TTS_BACKEND", "piper")
os.environ.setdefault("LIPSYNC_BACKEND", "wav2lip")
