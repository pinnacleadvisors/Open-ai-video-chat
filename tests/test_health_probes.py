import httpx
import pytest

from server.config import Settings
from server.health_probes import probe_lipsync_weights, probe_ollama, probe_tts_voice, run_probes


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(
        device="cpu",
        llm_backend="ollama",
        llm_model="llama3.1:8b",
        llm_base_url="http://does-not-exist:11434",
        tts_backend="piper",
        tts_voice="en_US-amy-medium",
        lipsync_backend="wav2lip",
        models_dir=tmp_path / "models",
    )
    return s


async def test_probe_ollama_unreachable_returns_false(settings):
    async with httpx.AsyncClient() as http:
        r = await probe_ollama(settings, http)
    assert r.name == "llm"
    assert r.ok is False
    assert "unreachable" in r.detail.lower() or "error" in r.detail.lower()


async def test_probe_ollama_skipped_for_openai_backend(settings):
    settings.llm_backend = "openai"
    async with httpx.AsyncClient() as http:
        r = await probe_ollama(settings, http)
    assert r.ok is True
    assert "openai" in r.detail


async def test_probe_ollama_succeeds_when_model_listed(settings):
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    )
    async with httpx.AsyncClient(transport=transport) as http:
        r = await probe_ollama(settings, http)
    assert r.ok is True


async def test_probe_ollama_fails_when_model_missing(settings):
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"models": [{"name": "other:7b"}]})
    )
    async with httpx.AsyncClient(transport=transport) as http:
        r = await probe_ollama(settings, http)
    assert r.ok is False
    assert "not pulled" in r.detail


async def test_probe_tts_voice_missing(settings):
    async with httpx.AsyncClient() as http:
        r = await probe_tts_voice(settings, http)
    assert r.ok is False


async def test_probe_tts_voice_present(settings, tmp_path):
    voice = settings.models_dir / "piper" / "en_US-amy-medium.onnx"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"x")
    async with httpx.AsyncClient() as http:
        r = await probe_tts_voice(settings, http)
    assert r.ok is True


async def test_probe_lipsync_weights_wav2lip_missing(settings):
    async with httpx.AsyncClient() as http:
        r = await probe_lipsync_weights(settings, http)
    assert r.ok is False
    assert "wav2lip" in r.detail


async def test_run_probes_aggregates_results(settings):
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    )
    async with httpx.AsyncClient(transport=transport) as http:
        report = await run_probes(settings, http, engines_ready=True, probes=(probe_ollama,))
    assert {c.name for c in report.checks} == {"engines", "llm"}
    assert report.ready is True


async def test_run_probes_not_ready_when_any_fails(settings):
    async with httpx.AsyncClient() as http:
        report = await run_probes(settings, http, engines_ready=False, probes=(probe_tts_voice,))
    assert report.ready is False
