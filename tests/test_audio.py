import numpy as np

from server.utils.audio import resample, s16_from_f32, to_mono_f32


def test_to_mono_f32_passthrough_float32():
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    out = to_mono_f32(x)
    assert out is x or np.array_equal(out, x)
    assert out.dtype == np.float32


def test_to_mono_f32_collapses_stereo():
    stereo = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    out = to_mono_f32(stereo)
    assert out.shape == (2,)
    assert np.allclose(out, 0.0)


def test_to_mono_f32_int16_to_float():
    x = np.array([0, 16384, -32768], dtype=np.int16)
    out = to_mono_f32(x)
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert out[1] > 0.4
    assert out[2] <= -0.999


def test_s16_from_f32_clips():
    x = np.array([0.0, 1.5, -1.5, 0.5], dtype=np.float32)
    out = s16_from_f32(x)
    assert out.dtype == np.int16
    assert out[1] == 32767
    assert out[2] == -32768


def test_resample_identity():
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = resample(x, 16000, 16000)
    assert np.array_equal(out, x)


def test_resample_changes_length_proportionally():
    sr_in, sr_out = 16000, 24000
    x = np.sin(np.linspace(0, 2 * np.pi * 440, sr_in)).astype(np.float32)
    out = resample(x, sr_in, sr_out)
    ratio = out.size / x.size
    assert 1.45 < ratio < 1.55
