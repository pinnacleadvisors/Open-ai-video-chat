"use client";

import { useEffect, useState } from "react";
import { VideoCall } from "@/components/VideoCall";
import { PersonaSetup } from "@/components/PersonaSetup";
import { fetchHealth, HealthInfo } from "@/lib/api";

export default function Page() {
  const [health, setHealth] = useState<HealthInfo | null>(null);

  useEffect(() => {
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

  return (
    <main className="min-h-screen px-6 py-8 max-w-6xl mx-auto">
      <header className="flex items-baseline justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold">open-ai-video-chat</h1>
          <p className="text-muted text-sm">self-hosted, open-source AI avatar</p>
        </div>
        {health ? (
          <div className="text-xs text-muted text-right space-y-0.5">
            <div>device: <span className="text-white">{health.device}</span></div>
            <div>llm: {health.llm.backend}/{health.llm.model}</div>
            <div>tts: {health.tts.backend}/{health.tts.voice}</div>
            <div>lipsync: {health.lipsync.backend} @ {health.lipsync.fps}fps</div>
          </div>
        ) : (
          <div className="text-xs text-red-400">server unreachable</div>
        )}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section><VideoCall /></section>
        <section><PersonaSetup /></section>
      </div>
    </main>
  );
}
