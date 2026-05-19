"use client";

import { useEffect, useState } from "react";
import { VideoCall } from "@/components/VideoCall";
import { PersonaSetup } from "@/components/PersonaSetup";
import { fetchHealth, HealthInfo, setAuthToken, getAuthToken } from "@/lib/api";

export default function Page() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [selected, setSelected] = useState<string | undefined>();
  const [tokenInput, setTokenInput] = useState("");
  const [tokenSet, setTokenSet] = useState(false);

  useEffect(() => {
    setTokenSet(!!getAuthToken());
    let alive = true;
    const tick = async () => {
      try {
        const h = await fetchHealth();
        if (alive) setHealth(h);
      } catch {
        if (alive) setHealth(null);
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const saveToken = () => {
    setAuthToken(tokenInput);
    setTokenSet(!!tokenInput);
    setTokenInput("");
  };

  return (
    <main className="min-h-screen px-6 py-8 max-w-6xl mx-auto">
      <header className="flex items-baseline justify-between mb-6 gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold">open-ai-video-chat</h1>
          <p className="text-muted text-sm">self-hosted, open-source AI avatar</p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          {!tokenSet && (
            <>
              <input
                placeholder="auth token (if required)"
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                className="bg-black/40 px-2 py-1 rounded border border-white/10 w-48"
              />
              <button onClick={saveToken} className="px-2 py-1 rounded bg-accent/80">
                set
              </button>
            </>
          )}
          {health ? (
            <div className="text-muted text-right space-y-0.5">
              <div>
                device: <span className="text-white">{health.device}</span>
              </div>
              <div>
                llm: {health.llm.backend}/{health.llm.model}
              </div>
              <div>
                tts: {health.tts.backend}/{health.tts.voice}
              </div>
              <div>
                lipsync: {health.lipsync.backend} @ {health.lipsync.fps}fps
              </div>
            </div>
          ) : (
            <div className="text-red-400">server unreachable</div>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section>
          <VideoCall personaId={selected} />
          {selected && (
            <p className="text-xs text-muted text-center mt-2">
              calling with persona <span className="text-white">{selected}</span>
            </p>
          )}
        </section>
        <section>
          <PersonaSetup selectedId={selected} onSelect={setSelected} />
        </section>
      </div>
    </main>
  );
}
