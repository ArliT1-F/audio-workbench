from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from argaudioworkbench.audio.loader import load_audio, to_mono


@dataclass
class ProcessingStep:
    name: str
    parameters: dict
    apply: Callable[[np.ndarray, int], tuple[np.ndarray, int]]


@dataclass
class ProjectState:
    audio_path: Path | None = None
    sample_rate: int | None = None
    original_sample_rate: int | None = None
    original_audio: np.ndarray | None = None
    working_audio: np.ndarray | None = None
    mono_view: bool = True
    history: list[ProcessingStep] = field(default_factory=list)
    redo_stack: list[ProcessingStep] = field(default_factory=list)

    def load_file(self, path: Path) -> None:
        data, samplerate = load_audio(path)
        self.audio_path = path
        self.sample_rate = samplerate
        self.original_sample_rate = samplerate
        self.original_audio = data
        self.working_audio = data
        self.history.clear()
        self.redo_stack.clear()

    def get_display_audio(self) -> np.ndarray:
        if self.working_audio is None:
            return np.array([], dtype=np.float32)
        if self.mono_view:
            return to_mono(self.working_audio)
        return self.working_audio

    def apply_step(self, step: ProcessingStep) -> None:
        if self.working_audio is None or self.sample_rate is None:
            return
        updated_audio, updated_rate = step.apply(self.working_audio, self.sample_rate)
        self.working_audio = updated_audio
        self.sample_rate = updated_rate
        self.history.append(step)
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.history or self.original_audio is None or self.original_sample_rate is None:
            return
        last_step = self.history.pop()
        self.redo_stack.append(last_step)
        audio = self.original_audio
        sample_rate = self.original_sample_rate
        for step in self.history:
            audio, sample_rate = step.apply(audio, sample_rate)
        self.working_audio = audio
        self.sample_rate = sample_rate

    def redo(self) -> None:
        if not self.redo_stack or self.working_audio is None or self.sample_rate is None:
            return
        step = self.redo_stack.pop()
        updated_audio, updated_rate = step.apply(self.working_audio, self.sample_rate)
        self.working_audio = updated_audio
        self.sample_rate = updated_rate
        self.history.append(step)

    def reset(self) -> None:
        if self.original_audio is None or self.original_sample_rate is None:
            return
        self.working_audio = self.original_audio
        self.sample_rate = self.original_sample_rate
        self.history.clear()
        self.redo_stack.clear()
