"use client";

import { useEffect, useRef, useState } from "react";
import { startCall, CallHandle } from "@/lib/webrtc";
import { connectTranscripts, TranscriptEvent } from "@/lib/api";

type ChatLine = { id: number; role: "user" | "assistant"; text: string };

export function VideoCall({ personaId }: { personaId?: string }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const callRef = useRef<CallHandle | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [calling, setCalling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chat, setChat] = useState<ChatLine[]>([]);
  const partial = useRef<string>("");
  const nextId = useRef(1);

  const append = (role: "user" | "assistant", text: string) => {
    setChat((c) => [...c, { id: nextId.current++, role, text }]);
  };

  const handleTranscript = (ev: TranscriptEvent) => {
    if (ev.role === "user" && ev.final) {
      append("user", ev.text);
    } else if (ev.role === "assistant") {
      if (!ev.final) {
        partial.current += ev.text;
      } else {
        append("assistant", ev.text || partial.current);
        partial.current = "";
      }
    }
  };

  const begin = async () => {
    setError(null);
    try {
      const call = await startCall({ personaId });
      callRef.current = call;
      if (videoRef.current) {
        videoRef.current.srcObject = call.remoteStream;
        await videoRef.current.play().catch(() => {});
      }
      wsRef.current = connectTranscripts(call.sessionId, handleTranscript);
      setCalling(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const end = async () => {
    if (callRef.current) await callRef.current.stop();
    callRef.current = null;
    if (wsRef.current) wsRef.current.close();
    wsRef.current = null;
    setCalling(false);
  };

  useEffect(() => () => { void end(); }, []);

  return (
    <div className="flex flex-col gap-4">
      <div className="relative aspect-square w-full max-w-[560px] mx-auto rounded-2xl overflow-hidden bg-panel border border-white/5">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full h-full object-cover"
        />
        {!calling && (
          <div className="absolute inset-0 flex items-center justify-center text-muted">
            press start to call
          </div>
        )}
      </div>

      <div className="flex items-center justify-center gap-3">
        {!calling ? (
          <button
            onClick={begin}
            className="px-6 py-2 rounded-full bg-accent text-white font-medium"
          >
            Start Call
          </button>
        ) : (
          <button
            onClick={end}
            className="px-6 py-2 rounded-full bg-red-500 text-white font-medium"
          >
            End Call
          </button>
        )}
      </div>

      {error && <p className="text-red-400 text-sm text-center">{error}</p>}

      <div className="max-w-[560px] mx-auto w-full max-h-72 overflow-y-auto rounded-xl bg-panel/60 border border-white/5 p-3 space-y-2 text-sm">
        {chat.length === 0 && (
          <p className="text-muted text-center">transcript will appear here</p>
        )}
        {chat.map((c) => (
          <div key={c.id} className={c.role === "user" ? "text-white" : "text-accent"}>
            <span className="opacity-60 mr-2">{c.role === "user" ? "you" : "avatar"}</span>
            {c.text}
          </div>
        ))}
      </div>
    </div>
  );
}
