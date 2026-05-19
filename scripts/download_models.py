#!/usr/bin/env python3
"""Download the open-source model checkpoints needed for the default pipeline.

Run after creating the venv. Idempotent.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "models" / "checkpoints"


# (url, dest_relative_to_checkpoints)
PIPER_VOICES = [
    ("https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
     "piper/en_US-amy-medium.onnx"),
    ("https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
     "piper/en_US-amy-medium.onnx.json"),
]

MUSETALK_FILES = [
    ("https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth",
     "musetalk/musetalkV15/unet.pth"),
    ("https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json",
     "musetalk/musetalkV15/musetalk.json"),
    ("https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e7e3e7ae7f2dbe14fa2c0c70b/tiny.pt",
     "musetalk/whisper/tiny.pt"),
]

WAV2LIP_FILES = [
    # ONNX export of Wav2Lip — useful as a CPU/GPU fallback.
    ("https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.onnx",
     "wav2lip/wav2lip.onnx"),
]


def fetch(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"  [skip] {dst.relative_to(ROOT)} ({dst.stat().st_size//1024} KB)")
        return
    print(f"  [get ] {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
            total = int(r.headers.get("Content-Length", "0") or 0)
            done = 0
            chunk = 1 << 20
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(f"\r        {pct:5.1f}%  ({done//1048576} / {total//1048576} MB)", end="", flush=True)
            print()
        tmp.replace(dst)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"  [warn] failed: {e}")


def main() -> int:
    print(f"checkpoint dir: {CHECKPOINTS}")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    print("\n-- piper voices --")
    for url, rel in PIPER_VOICES:
        fetch(url, CHECKPOINTS / rel)

    if os.environ.get("OAVC_SKIP_MUSETALK") != "1":
        print("\n-- musetalk --")
        for url, rel in MUSETALK_FILES:
            fetch(url, CHECKPOINTS / rel)

    if os.environ.get("OAVC_SKIP_WAV2LIP") != "1":
        print("\n-- wav2lip (fallback) --")
        for url, rel in WAV2LIP_FILES:
            fetch(url, CHECKPOINTS / rel)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
