from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class SpectrogramSettings:
    n_fft: int = 2048
    hop_length: int = 512
    window: str = "hann"
    db_range: float = 80.0
    log_frequency: bool = False
    max_frequency: float | None = None


@dataclass(frozen=True)
class SpectrogramResult:
    frequencies: np.ndarray
    times: np.ndarray
    magnitude_db: np.ndarray


def compute_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    window: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if audio.ndim != 1:
        raise ValueError("Audio must be mono for STFT computation.")
    if audio.size == 0:
        return np.array([]), np.array([]), np.empty((0, 0), dtype=np.complex64)

    n_fft = min(n_fft, audio.size)
    hop_length = min(hop_length, n_fft)
    hop_length = max(hop_length, 1)

    window_fn = signal.get_window(window, n_fft, fftbins=True)
    frequencies, times, stft_matrix = signal.stft(
        audio,
        fs=sample_rate,
        window=window_fn,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False,
    )
    return frequencies, times, stft_matrix


def magnitude_to_db(magnitude: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    magnitude = np.maximum(magnitude, amin)
    return 20.0 * np.log10(magnitude / ref)


def render_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    settings: SpectrogramSettings,
) -> SpectrogramResult:
    frequencies, times, stft_matrix = compute_stft(
        audio,
        sample_rate,
        settings.n_fft,
        settings.hop_length,
        settings.window,
    )
    magnitude = np.abs(stft_matrix)
    magnitude_db = magnitude_to_db(magnitude, ref=np.max(magnitude) if magnitude.size else 1.0)

    if settings.max_frequency is not None:
        freq_mask = frequencies <= settings.max_frequency
        frequencies = frequencies[freq_mask]
        magnitude_db = magnitude_db[freq_mask, :]

    if settings.log_frequency and frequencies.size > 1:
        min_freq = max(frequencies[1], 1.0)
        log_freqs = np.geomspace(min_freq, frequencies[-1], num=frequencies.size)
        remapped = np.empty((log_freqs.size, magnitude_db.shape[1]), dtype=magnitude_db.dtype)
        for idx in range(magnitude_db.shape[1]):
            remapped[:, idx] = np.interp(
                log_freqs,
                frequencies,
                magnitude_db[:, idx],
                left=magnitude_db.min(),
                right=magnitude_db.min(),
            )
        magnitude_db = remapped
        frequencies = log_freqs

    return SpectrogramResult(frequencies=frequencies, times=times, magnitude_db=magnitude_db)
