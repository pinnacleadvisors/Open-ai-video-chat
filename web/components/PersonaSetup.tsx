"use client";

import { useState } from "react";
import { uploadPersonaImage, uploadVoiceSample, updatePersona } from "@/lib/api";

export function PersonaSetup() {
  const [imageOk, setImageOk] = useState(false);
  const [voiceOk, setVoiceOk] = useState(false);
  const [prompt, setPrompt] = useState(
    "You are a friendly, concise video-call assistant. Speak naturally, like a person.",
  );
  const [status, setStatus] = useState<string | null>(null);

  const onImage = async (f: File | null) => {
    if (!f) return;
    setStatus("uploading portrait…");
    try {
      await uploadPersonaImage(f);
      setImageOk(true);
      setStatus("portrait ready");
    } catch (e: any) {
      setStatus(`image upload failed: ${e.message ?? e}`);
    }
  };

  const onVoice = async (f: File | null) => {
    if (!f) return;
    setStatus("uploading voice sample…");
    try {
      const { path } = await uploadVoiceSample(f);
      await updatePersona({ speaker_wav: path });
      setVoiceOk(true);
      setStatus("voice clone ready");
    } catch (e: any) {
      setStatus(`voice upload failed: ${e.message ?? e}`);
    }
  };

  const savePrompt = async () => {
    setStatus("saving system prompt…");
    try {
      await updatePersona({ system_prompt: prompt });
      setStatus("system prompt saved");
    } catch (e: any) {
      setStatus(`save failed: ${e.message ?? e}`);
    }
  };

  return (
    <div className="rounded-2xl bg-panel border border-white/5 p-5 space-y-4 max-w-[560px] mx-auto w-full">
      <h2 className="text-lg font-semibold">Persona</h2>

      <label className="block space-y-1">
        <span className="text-sm text-muted">Portrait (JPG/PNG) {imageOk && "✓"}</span>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => onImage(e.target.files?.[0] ?? null)}
          className="block w-full text-sm"
        />
      </label>

      <label className="block space-y-1">
        <span className="text-sm text-muted">Voice sample (6–30s WAV/MP3, optional clone) {voiceOk && "✓"}</span>
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => onVoice(e.target.files?.[0] ?? null)}
          className="block w-full text-sm"
        />
      </label>

      <label className="block space-y-1">
        <span className="text-sm text-muted">System prompt</span>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full bg-black/40 rounded-md p-2 text-sm h-24 border border-white/10"
        />
        <button
          onClick={savePrompt}
          className="mt-1 px-3 py-1 rounded-md bg-accent/80 text-white text-sm"
        >
          Save prompt
        </button>
      </label>

      {status && <p className="text-xs text-muted">{status}</p>}
    </div>
  );
}
