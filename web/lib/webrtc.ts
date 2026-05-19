export type CallHandle = {
  pc: RTCPeerConnection;
  sessionId: string;
  remoteStream: MediaStream;
  stop: () => Promise<void>;
};

export async function startCall(): Promise<CallHandle> {
  const pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
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

  // Receive avatar audio + video
  pc.addTransceiver("video", { direction: "recvonly" });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  // Wait for ICE gathering to finish (trickle disabled for simplicity)
  await waitForIceComplete(pc);

  const resp = await fetch("/api/webrtc/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription!.sdp, type: pc.localDescription!.type }),
  });
  if (!resp.ok) throw new Error(`offer rejected: ${resp.status}`);
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
        await fetch(`/api/webrtc/${answer.id}`, { method: "DELETE" });
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
