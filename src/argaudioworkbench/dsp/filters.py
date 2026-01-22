from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FilterSettings:
    low_hz: float | None = None
    high_hz: float | None = None
    order: int = 4


@dataclass(frozen=True)
class NotchSettings:
    center_hz: float
    q: float = 30.0


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak == 0:
        return audio.copy()
    return audio / peak


def reverse(audio: np.ndarray) -> np.ndarray:
    return audio[::-1].copy()


def trim(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float) -> np.ndarray:
    start_idx = max(int(start_s * sample_rate), 0)
    end_idx = min(int(end_s * sample_rate), audio.shape[0])
    if start_idx >= end_idx:
        return np.array([], dtype=audio.dtype)
    return audio[start_idx:end_idx].copy()


def resample(audio: np.ndarray, sample_rate: int, target_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate == target_rate:
        return audio.copy(), sample_rate
    duration = audio.shape[0] / sample_rate
    target_samples = int(duration * target_rate)
    if audio.ndim == 1:
        resampled = signal.resample(audio, target_samples)
    else:
        resampled = np.vstack([signal.resample(audio[:, ch], target_samples) for ch in range(audio.shape[1])]).T
    return resampled.astype(audio.dtype, copy=False), target_rate


def bandpass(audio: np.ndarray, sample_rate: int, settings: FilterSettings) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    low = (settings.low_hz or 0.0) / nyquist
    high = (settings.high_hz or nyquist) / nyquist
    if low <= 0 and high >= 1:
        return audio.copy()
    if low <= 0:
        sos = signal.butter(settings.order, high, btype="low", output="sos")
    elif high >= 1:
        sos = signal.butter(settings.order, low, btype="high", output="sos")
    else:
        sos = signal.butter(settings.order, [low, high], btype="band", output="sos")
    if audio.ndim == 1:
        return signal.sosfiltfilt(sos, audio)
    filtered = np.vstack([signal.sosfiltfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])]).T
    return filtered


def notch(audio: np.ndarray, sample_rate: int, settings: NotchSettings) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    w0 = settings.center_hz / nyquist
    if w0 <= 0 or w0 >= 1:
        return audio.copy()
    b, a = signal.iirnotch(w0, settings.q)
    if audio.ndim == 1:
        return signal.filtfilt(b, a, audio)
    filtered = np.vstack([signal.filtfilt(b, a, audio[:, ch]) for ch in range(audio.shape[1])]).T
    return filtered
