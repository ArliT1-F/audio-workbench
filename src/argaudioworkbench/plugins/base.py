from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PySide6 import QtCore


@dataclass(frozen=True)
class ToolResult:
    command: str
    stdout: str
    stderr: str
    return_code: int
    output_dir: Path


class PluginInterface(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def description(self) -> str:
        ...


class ToolWorker(QtCore.QThread):
    finished_with_result = QtCore.Signal(ToolResult)

    def __init__(self, command: list[str], output_dir: Path, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._command = command
        self._output_dir = output_dir

    def run(self) -> None:
        import subprocess

        self._output_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.run(self._command, capture_output=True, text=True)
        result = ToolResult(
            command=" ".join(self._command),
            stdout=process.stdout,
            stderr=process.stderr,
            return_code=process.returncode,
            output_dir=self._output_dir,
        )
        log_path = self._output_dir / "tool_run.log"
        log_path.write_text(
            f"$ {result.command}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )
        self.finished_with_result.emit(result)
