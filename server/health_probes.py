from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from .config import Settings


@dataclass
class ProbeResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ReadinessReport:
    ready: bool
    checks: list[ProbeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
        }


Probe = Callable[[Settings, httpx.AsyncClient], Awaitable[ProbeResult]]


async def probe_gpu(settings: Settings, _http: httpx.AsyncClient) -> ProbeResult:
    if settings.device != "cuda":
        return ProbeResult("gpu", True, f"device={settings.device}")
    try:
        import torch
    except ImportError:
        return ProbeResult("gpu", False, "torch not installed")
    if not torch.cuda.is_available():
        return ProbeResult("gpu", False, "torch.cuda.is_available() == False")
    return ProbeResult("gpu", True, f"{torch.cuda.device_count()} device(s)")


async def probe_ollama(settings: Settings, http: httpx.AsyncClient) -> ProbeResult:
    if settings.llm_backend != "ollama":
        return ProbeResult("llm", True, f"backend={settings.llm_backend}")
    url = settings.llm_base_url.rstrip("/") + "/api/tags"
    try:
        r = await http.get(url, timeout=1.5)
        r.raise_for_status()
    except Exception as e:
        return ProbeResult("llm", False, f"ollama unreachable: {e!s}"[:200])
    models = {m.get("name") for m in (r.json().get("models") or [])}
    if settings.llm_model not in models and f"{settings.llm_model}:latest" not in models:
        return ProbeResult(
            "llm",
            False,
            f"model {settings.llm_model!r} not pulled. run: ollama pull {settings.llm_model}",
        )
    return ProbeResult("llm", True, f"ollama ok, {len(models)} model(s)")


async def probe_lipsync_weights(settings: Settings, _http: httpx.AsyncClient) -> ProbeResult:
    if settings.lipsync_backend == "musetalk":
        unet = settings.models_dir / "musetalk" / "musetalkV15" / "unet.pth"
        if not unet.exists():
            return ProbeResult("lipsync", False, f"missing {unet} (run scripts/setup.sh)")
        return ProbeResult("lipsync", True, "musetalk weights present")
    if settings.lipsync_backend == "wav2lip":
        onnx = settings.models_dir / "wav2lip" / "wav2lip.onnx"
        if not onnx.exists():
            return ProbeResult("lipsync", False, f"missing {onnx} (run scripts/setup.sh)")
        return ProbeResult("lipsync", True, "wav2lip weights present")
    return ProbeResult("lipsync", True, f"backend={settings.lipsync_backend}")


async def probe_tts_voice(settings: Settings, _http: httpx.AsyncClient) -> ProbeResult:
    if settings.tts_backend != "piper":
        return ProbeResult("tts", True, f"backend={settings.tts_backend}")
    voice = settings.models_dir / "piper" / f"{settings.tts_voice}.onnx"
    if not voice.exists():
        return ProbeResult("tts", False, f"piper voice missing: {voice}")
    return ProbeResult("tts", True, f"voice={settings.tts_voice}")


def _engines_loaded_probe(engines_ready: bool) -> Probe:
    async def probe(_s, _h) -> ProbeResult:
        return ProbeResult("engines", engines_ready, "loaded" if engines_ready else "not loaded yet")
    return probe


DEFAULT_PROBES: tuple[Probe, ...] = (
    probe_gpu,
    probe_ollama,
    probe_lipsync_weights,
    probe_tts_voice,
)


async def run_probes(
    settings: Settings,
    http: httpx.AsyncClient,
    *,
    engines_ready: bool,
    probes: tuple[Probe, ...] = DEFAULT_PROBES,
) -> ReadinessReport:
    all_probes = (_engines_loaded_probe(engines_ready), *probes)
    results = await asyncio.gather(*(p(settings, http) for p in all_probes))
    ready = all(r.ok for r in results)
    return ReadinessReport(ready=ready, checks=list(results))
