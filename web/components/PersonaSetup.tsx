"use client";

import { useEffect, useState } from "react";
import {
  createPersona,
  deletePersona,
  fetchVoices,
  listPersonas,
  uploadVoiceSample,
  Persona,
} from "@/lib/api";

export function PersonaSetup({
  selectedId,
  onSelect,
}: {
  selectedId?: string;
  onSelect: (id: string | undefined) => void;
}) {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [voices, setVoices] = useState<string[]>([]);
  const [name, setName] = useState("My Avatar");
  const [voice, setVoice] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [speakerWav, setSpeakerWav] = useState<string | undefined>();
  const [prompt, setPrompt] = useState(
    "You are a friendly, concise video-call assistant. Speak naturally, like a person.",
  );
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      const [v, p] = await Promise.all([fetchVoices(), listPersonas()]);
      setVoices(v.voices);
      setPersonas(p);
      if (!voice && v.voices[0]) setVoice(v.voices[0]);
    } catch (e: unknown) {
      setStatus(`load failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const onVoiceClone = async (f: File | null) => {
    if (!f) return;
    setBusy(true);
    setStatus("uploading voice sample…");
    try {
      const { path } = await uploadVoiceSample(f);
      setSpeakerWav(path);
      setStatus("voice clone uploaded");
    } catch (e: unknown) {
      setStatus(`voice upload failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const submit = async () => {
    if (!image) {
      setStatus("portrait required");
      return;
    }
    setBusy(true);
    setStatus("creating persona…");
    try {
      const p = await createPersona({
        name,
        voice,
        image,
        speaker_wav: speakerWav,
        system_prompt: prompt,
      });
      setStatus("persona created");
      await refresh();
      onSelect(p.id);
    } catch (e: unknown) {
      setStatus(`create failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    await deletePersona(id);
    if (selectedId === id) onSelect(undefined);
    await refresh();
  };

  return (
    <div className="rounded-2xl bg-panel border border-white/5 p-5 space-y-5 max-w-[560px] mx-auto w-full">
      <div className="space-y-2">
        <h2 className="text-lg font-semibold">Personas</h2>
        {personas.length === 0 && <p className="text-muted text-sm">no personas yet</p>}
        <ul className="space-y-1">
          {personas.map((p) => (
            <li
              key={p.id}
              className={`flex items-center justify-between rounded-lg px-3 py-2 cursor-pointer ${
                selectedId === p.id ? "bg-accent/20 border border-accent/40" : "hover:bg-white/5"
              }`}
              onClick={() => onSelect(p.id)}
            >
              <div>
                <div className="text-sm">{p.name}</div>
                <div className="text-xs text-muted">{p.voice}</div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  void remove(p.id);
                }}
                className="text-xs text-muted hover:text-red-400"
              >
                delete
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="border-t border-white/5 pt-4 space-y-3">
        <h3 className="text-sm font-medium">Create new</h3>

        <label className="block space-y-1">
          <span className="text-xs text-muted">Name</span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full bg-black/40 rounded-md p-2 text-sm border border-white/10"
          />
        </label>

        <label className="block space-y-1">
          <span className="text-xs text-muted">Voice</span>
          <select
            value={voice}
            onChange={(e) => setVoice(e.target.value)}
            className="w-full bg-black/40 rounded-md p-2 text-sm border border-white/10"
          >
            {voices.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>

        <label className="block space-y-1">
          <span className="text-xs text-muted">Portrait (PNG/JPG/WebP)</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImage(e.target.files?.[0] ?? null)}
            className="block w-full text-sm"
          />
        </label>

        <label className="block space-y-1">
          <span className="text-xs text-muted">
            Voice sample (optional, 6–30s WAV/MP3 for cloning)
          </span>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => onVoiceClone(e.target.files?.[0] ?? null)}
            className="block w-full text-sm"
          />
        </label>

        <label className="block space-y-1">
          <span className="text-xs text-muted">System prompt</span>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full bg-black/40 rounded-md p-2 text-sm h-20 border border-white/10"
          />
        </label>

        <button
          onClick={submit}
          disabled={busy || !image}
          className="px-4 py-2 rounded-md bg-accent text-white text-sm disabled:opacity-50"
        >
          Create persona
        </button>
      </div>

      {status && <p className="text-xs text-muted">{status}</p>}
    </div>
  );
}
