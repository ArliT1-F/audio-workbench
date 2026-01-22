from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


class AudioLoadError(RuntimeError):
    pass


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise AudioLoadError(f"File not found: {path}")

    if path.suffix.lower() == ".mp3":
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise AudioLoadError(
                "MP3 support requires pydub and ffmpeg. Install with `pip install -e .[mp3]`."
            ) from exc
        audio_segment = AudioSegment.from_file(path)
        samplerate = audio_segment.frame_rate
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        if audio_segment.channels > 1:
            samples = samples.reshape((-1, audio_segment.channels))
        else:
            samples = samples.reshape((-1, 1))
        data = samples / np.iinfo(audio_segment.array_type).max
    else:
        try:
            data, samplerate = sf.read(str(path), always_2d=True)
        except RuntimeError as exc:
            raise AudioLoadError(f"Could not read audio file: {exc}") from exc

    if data.size == 0:
        raise AudioLoadError("Audio file is empty.")

    return data, samplerate


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError("Expected audio array with shape (samples, channels).")
    return np.mean(audio, axis=1)


def get_channel(audio: np.ndarray, channel: int) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if channel < 0 or channel >= audio.shape[1]:
        raise ValueError("Invalid channel index.")
    return audio[:, channel]
