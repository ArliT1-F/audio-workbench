import numpy as np

from argaudioworkbench.dsp.stft import SpectrogramSettings, compute_stft, render_spectrogram


def test_compute_stft_shape() -> None:
    sample_rate = 8000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)
    freqs, times, stft = compute_stft(audio, sample_rate, n_fft=1024, hop_length=256, window="hann")
    assert freqs.size > 0
    assert times.size > 0
    assert stft.shape[0] == freqs.size


def test_render_spectrogram_log_frequency() -> None:
    sample_rate = 8000
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = np.sin(2 * np.pi * 1000 * t)
    settings = SpectrogramSettings(n_fft=1024, hop_length=256, window="hann", log_frequency=True)
    result = render_spectrogram(audio, sample_rate, settings)
    assert result.magnitude_db.shape[0] == result.frequencies.size
