from __future__ import annotations

import shutil
from pathlib import Path

from PySide6 import QtCore

from argaudioworkbench.plugins.base import ToolWorker


class MinimodemPlugin(QtCore.QObject):
    name = "minimodem"

    def description(self) -> str:
        return "Decode RTTY/FSK via minimodem if installed."

    def is_available(self) -> bool:
        return shutil.which("minimodem") is not None

    def decode(self, wav_path: Path, output_dir: Path, args: list[str]) -> ToolWorker | None:
        if not self.is_available():
            return None
        command = ["minimodem", *args, str(wav_path)]
        return ToolWorker(command, output_dir)
