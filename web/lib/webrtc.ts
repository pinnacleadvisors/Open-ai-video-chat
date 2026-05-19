import { getAuthToken } from "./api";

export type CallHandle = {
  pc: RTCPeerConnection;
  sessionId: string;
  remoteStream: MediaStream;
  stop: () => Promise<void>;
};

export type CallOptions = {
  personaId?: string;
  iceServers?: RTCIceServer[];
};

export async function startCall(opts: CallOptions = {}): Promise<CallHandle> {
  const pc = new RTCPeerConnection({
    iceServers: opts.iceServers ?? [{ urls: "stun:stun.l.google.com:19302" }],
  });

  const remoteStream = new MediaStream();
  pc.ontrack = (e) => {
    e.streams[0].getTracks().forEach((t) => remoteStream.addTrack(t));
  };

  const mic = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
      sampleRate: 48000,
    },
    video: false,
  });
  mic.getAudioTracks().forEach((t) => pc.addTrack(t, mic));

  pc.addTransceiver("video", { direction: "recvonly" });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitForIceComplete(pc);

  const token = getAuthToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const resp = await fetch("/api/webrtc/offer", {
    method: "POST",
    headers,
    body: JSON.stringify({
      sdp: pc.localDescription!.sdp,
      type: pc.localDescription!.type,
      persona_id: opts.personaId,
    }),
  });
  if (!resp.ok) throw new Error(`offer rejected: ${resp.status} ${await resp.text()}`);
  const answer = await resp.json();
  await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });

  return {
    pc,
    sessionId: answer.id,
    remoteStream,
    stop: async () => {
      mic.getTracks().forEach((t) => t.stop());
      pc.close();
      try {
        await fetch(`/api/webrtc/${answer.id}`, { method: "DELETE", headers });
      } catch {
        // ignore
      }
    },
  };
}

function waitForIceComplete(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 2000);
  });
}
