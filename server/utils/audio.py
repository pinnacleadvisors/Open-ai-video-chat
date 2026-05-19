from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resampling with scipy. Mono float32."""
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def to_mono_f32(pcm: np.ndarray) -> np.ndarray:
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=-1)
    if pcm.dtype != np.float32:
        if np.issubdtype(pcm.dtype, np.integer):
            max_v = float(np.iinfo(pcm.dtype).max)
            pcm = pcm.astype(np.float32) / max_v
        else:
            pcm = pcm.astype(np.float32)
    return pcm


def s16_from_f32(pcm: np.ndarray) -> np.ndarray:
    return np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
