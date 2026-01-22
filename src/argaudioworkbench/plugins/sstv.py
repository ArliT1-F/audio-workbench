from __future__ import annotations

import shutil
from pathlib import Path

from PySide6 import QtCore

from argaudioworkbench.plugins.base import ToolWorker


class SSTVPlugin(QtCore.QObject):
    name = "SSTV"

    def description(self) -> str:
        return "SSTV decoder helper using qsstv or sstv CLI if installed."

    def is_available(self) -> bool:
        return shutil.which("qsstv") is not None or shutil.which("sstv") is not None

    def available_modes(self) -> list[str]:
        modes = []
        if shutil.which("qsstv"):
            modes.append("Launch QSSTV")
        if shutil.which("sstv"):
            modes.append("Decode with sstv")
        return modes

    def launch_qsstv(self) -> None:
        if shutil.which("qsstv"):
            QtCore.QProcess.startDetached("qsstv")

    def decode_with_sstv(self, wav_path: Path, output_dir: Path) -> ToolWorker | None:
        if not shutil.which("sstv"):
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "sstv.png"
        command = ["sstv", "-i", str(wav_path), "-o", str(output_path)]
        return ToolWorker(command, output_dir)
