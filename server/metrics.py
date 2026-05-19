from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


class _Metrics:
    def __init__(self) -> None:
        self.sessions_active = Gauge(
            "oavc_sessions_active", "Number of active conversational sessions"
        )
        self.utterances_total = Counter(
            "oavc_utterances_total", "User utterances transcribed"
        )
        self.barge_ins_total = Counter(
            "oavc_barge_ins_total", "Barge-in interruptions"
        )
        self.llm_ttft_seconds = Histogram(
            "oavc_llm_ttft_seconds",
            "Time from user utterance commit to first LLM token",
            buckets=(0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4),
        )
        self.avatar_render_seconds = Histogram(
            "oavc_avatar_render_seconds",
            "Time spent rendering a single avatar AVPair",
            buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
        )
        self.webrtc_negotiations_total = Counter(
            "oavc_webrtc_negotiations_total", "WebRTC SDP negotiations"
        )
        self.webrtc_negotiation_failures_total = Counter(
            "oavc_webrtc_negotiation_failures_total", "Failed WebRTC negotiations"
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST


metrics = _Metrics()
