export type HealthInfo = {
  status: string;
  device: string;
  llm: { backend: string; model: string };
  tts: { backend: string; voice: string };
  lipsync: { backend: string; fps: number };
  stt: { model: string };
  sessions: number;
};

export type Persona = {
  id: string;
  name: string;
  image_path: string;
  voice: string;
  speaker_wav?: string | null;
  system_prompt?: string | null;
  created_at: number;
};

export type Voices = { backend: string; voices: string[] };
export type TranscriptEvent = { role: "user" | "assistant"; text: string; final: boolean };

const TOKEN_KEY = "oavc.token";

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const token = typeof window !== "undefined" ? localStorage.getItem(TOKEN_KEY) : null;
  const h: Record<string, string> = { ...extra };
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

async function jsonOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export async function fetchHealth(): Promise<HealthInfo> {
  return jsonOrThrow(await fetch("/api/health", { headers: headers() }));
}

export async function fetchVoices(): Promise<Voices> {
  return jsonOrThrow(await fetch("/api/voices", { headers: headers() }));
}

export async function listPersonas(): Promise<Persona[]> {
  return jsonOrThrow(await fetch("/api/personas", { headers: headers() }));
}

export async function createPersona(opts: {
  name: string;
  voice: string;
  image: File;
  speaker_wav?: string;
  system_prompt?: string;
}): Promise<Persona> {
  const params = new URLSearchParams({ name: opts.name, voice: opts.voice });
  if (opts.speaker_wav) params.set("speaker_wav", opts.speaker_wav);
  if (opts.system_prompt) params.set("system_prompt", opts.system_prompt);
  const fd = new FormData();
  fd.append("file", opts.image);
  return jsonOrThrow(
    await fetch(`/api/personas?${params.toString()}`, {
      method: "POST",
      headers: headers(),
      body: fd,
    }),
  );
}

export async function deletePersona(id: string): Promise<void> {
  await fetch(`/api/personas/${id}`, { method: "DELETE", headers: headers() });
}

export async function uploadVoiceSample(file: File): Promise<{ path: string }> {
  const fd = new FormData();
  fd.append("file", file);
  return jsonOrThrow(
    await fetch("/api/persona/voice", { method: "POST", headers: headers(), body: fd }),
  );
}

export function connectTranscripts(
  sessionId: string,
  onEvent: (ev: TranscriptEvent) => void,
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const token = localStorage.getItem(TOKEN_KEY);
  const qs = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(
    `${proto}://${window.location.host}/ws/transcripts/${encodeURIComponent(sessionId)}${qs}`,
  );
  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch {
      // ignore
    }
  };
  return ws;
}

export function setAuthToken(token: string): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export function getAuthToken(): string {
  return typeof window !== "undefined" ? (localStorage.getItem(TOKEN_KEY) ?? "") : "";
}
