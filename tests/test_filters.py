import numpy as np

from argaudioworkbench.dsp.filters import FilterSettings, NotchSettings, bandpass, normalize, notch, reverse


def test_normalize_peak() -> None:
    audio = np.array([0.0, 0.5, -0.25], dtype=np.float32)
    normalized = normalize(audio)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)


def test_reverse() -> None:
    audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reversed_audio = reverse(audio)
    assert np.array_equal(reversed_audio, np.array([3.0, 2.0, 1.0], dtype=np.float32))


def test_bandpass_noop() -> None:
    audio = np.random.randn(1024).astype(np.float32)
    settings = FilterSettings(low_hz=None, high_hz=None, order=4)
    filtered = bandpass(audio, 8000, settings)
    assert np.allclose(filtered, audio, atol=1e-3)


def test_notch_invalid_range() -> None:
    audio = np.random.randn(1024).astype(np.float32)
    settings = NotchSettings(center_hz=100000, q=30.0)
    filtered = notch(audio, 8000, settings)
    assert np.allclose(filtered, audio)
