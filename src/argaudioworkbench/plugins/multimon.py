from __future__ import annotations

import shutil
from pathlib import Path

from PySide6 import QtCore

from argaudioworkbench.plugins.base import ToolWorker


class MultimonPlugin(QtCore.QObject):
    name = "multimon-ng"

    def description(self) -> str:
        return "Decode DTMF/AFSK/FSK using multimon-ng if installed."

    def is_available(self) -> bool:
        return shutil.which("multimon-ng") is not None

    def decode(self, wav_path: Path, output_dir: Path, modes: list[str]) -> ToolWorker | None:
        if not self.is_available():
            return None
        command = ["multimon-ng", "-a", ",".join(modes), str(wav_path)]
        return ToolWorker(command, output_dir)
