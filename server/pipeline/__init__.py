# Re-exports for type-checking only. Submodules with heavy dependencies
# (cv2, torch, mediapipe) are imported on demand to keep `server.pipeline.llm`
# and `server.pipeline.tts` importable in lightweight environments (tests, CI).

from typing import TYPE_CHECKING

__all__ = [
    "STTSession",
    "Utterance",
    "VADStateMachine",
    "LLMSession",
    "ChatState",
    "Message",
    "TTSSession",
    "AudioChunk",
    "Phraser",
    "AvatarSession",
    "AVPair",
    "VideoFrame",
    "Orchestrator",
    "TranscriptEvent",
]


def __getattr__(name: str):
    if name in ("STTSession", "Utterance", "VADStateMachine"):
        from . import stt as _m
        return getattr(_m, name)
    if name in ("LLMSession", "ChatState", "Message"):
        from . import llm as _m
        return getattr(_m, name)
    if name in ("TTSSession", "AudioChunk", "Phraser"):
        from . import tts as _m
        return getattr(_m, name)
    if name == "AvatarSession":
        from . import avatar as _m
        return _m.AvatarSession
    if name in ("AVPair", "VideoFrame"):
        from . import avatar_backends as _m
        return getattr(_m, name)
    if name in ("Orchestrator", "TranscriptEvent"):
        from . import orchestrator as _m
        return getattr(_m, name)
    raise AttributeError(name)


if TYPE_CHECKING:
    from .avatar import AvatarSession  # noqa: F401
    from .avatar_backends import AVPair, VideoFrame  # noqa: F401
    from .llm import ChatState, LLMSession, Message  # noqa: F401
    from .orchestrator import Orchestrator, TranscriptEvent  # noqa: F401
    from .stt import STTSession, Utterance, VADStateMachine  # noqa: F401
    from .tts import AudioChunk, Phraser, TTSSession  # noqa: F401
