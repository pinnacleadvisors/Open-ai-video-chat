export type HealthInfo = {
  status: string;
  device: string;
  llm: { backend: string; model: string };
  tts: { backend: string; voice: string };
  lipsync: { backend: string; fps: number };
  stt: { model: string };
  sessions: number;
};

export async function fetchHealth(): Promise<HealthInfo> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function uploadPersonaImage(file: File): Promise<{ path: string }> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/persona/image", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`persona image upload failed: ${r.status}`);
  return r.json();
}

export async function uploadVoiceSample(file: File): Promise<{ path: string }> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/persona/voice", { method: "POST", body: fd });
  if (!r.ok) throw new Error(`voice upload failed: ${r.status}`);
  return r.json();
}

export async function updatePersona(config: {
  system_prompt?: string;
  voice?: string;
  speaker_wav?: string;
}): Promise<void> {
  const r = await fetch("/api/persona/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!r.ok) throw new Error(`persona config failed: ${r.status}`);
}

export type TranscriptEvent = { role: "user" | "assistant"; text: string; final: boolean };

export function connectTranscripts(onEvent: (ev: TranscriptEvent) => void): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/transcripts`);
  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch {
      // ignore
    }
  };
  return ws;
}
