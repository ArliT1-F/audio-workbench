from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401
from PySide6 import QtCore, QtGui, QtWidgets

from argaudioworkbench.core.autoscan import run_autoscan
from argaudioworkbench.core.evidence import EvidenceMetadata, hash_audio, save_evidence_bundle
from argaudioworkbench.core.project import ProcessingStep, ProjectState
from argaudioworkbench.dsp.filters import FilterSettings, NotchSettings, bandpass, normalize, notch, resample, reverse, trim
from argaudioworkbench.dsp.stft import SpectrogramSettings
from argaudioworkbench.plugins.minimodem import MinimodemPlugin
from argaudioworkbench.plugins.multimon import MultimonPlugin
from argaudioworkbench.plugins.sstv import SSTVPlugin
from argaudioworkbench.ui.spectrogram_widget import SpectrogramWidget
from argaudioworkbench.ui.waveform_widget import WaveformWidget


class AutoscanWorker(QtCore.QThread):
    finished_with_manifest = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, audio: np.ndarray, sample_rate: int, output_dir: Path) -> None:
        super().__init__()
        self._audio = audio
        self._sample_rate = sample_rate
        self._output_dir = output_dir

    def run(self) -> None:
        try:
            manifest = run_autoscan(self._audio, self._sample_rate, self._output_dir)
        except Exception as exc:  # noqa: BLE001 - show errors to user
            self.failed.emit(str(exc))
            return
        self.finished_with_manifest.emit(manifest)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ArgAudioWorkbench")
        self.resize(1400, 900)

        self.project = ProjectState()
        self.sstv_plugin = SSTVPlugin()
        self.multimon_plugin = MultimonPlugin()
        self.minimodem_plugin = MinimodemPlugin()

        self._build_ui()
        self._connect_actions()

    def _build_ui(self) -> None:
        self._setup_actions()
        self._setup_menu()

        central_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        top_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.control_panel = self._build_control_panel()
        top_split.addWidget(self.control_panel)

        view_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.waveform = WaveformWidget()
        self.spectrogram = SpectrogramWidget()
        view_split.addWidget(self.waveform)
        view_split.addWidget(self.spectrogram)
        view_split.setStretchFactor(0, 1)
        view_split.setStretchFactor(1, 2)

        top_split.addWidget(view_split)
        top_split.setStretchFactor(1, 3)
        central_split.addWidget(top_split)

        self.log_panel = QtWidgets.QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        central_split.addWidget(self.log_panel)
        central_split.setStretchFactor(0, 4)
        central_split.setStretchFactor(1, 1)

        self.setCentralWidget(central_split)

        self.status_bar = self.statusBar()

    def _setup_actions(self) -> None:
        self.open_action = QtGui.QAction("Open", self)
        self.open_action.setShortcut("Ctrl+O")
        self.export_action = QtGui.QAction("Export Spectrogram", self)
        self.export_action.setShortcut("Ctrl+E")
        self.autoscan_action = QtGui.QAction("Autoscan", self)
        self.evidence_action = QtGui.QAction("Evidence Bundle", self)
        self.undo_action = QtGui.QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.redo_action = QtGui.QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")

    def _setup_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.autoscan_action)
        file_menu.addAction(self.evidence_action)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.file_label = QtWidgets.QLabel("No file loaded")
        layout.addWidget(self.file_label)

        self.mono_toggle = QtWidgets.QCheckBox("Mono view")
        self.mono_toggle.setChecked(True)
        layout.addWidget(self.mono_toggle)

        spectro_group = QtWidgets.QGroupBox("Spectrogram Settings")
        spectro_layout = QtWidgets.QFormLayout(spectro_group)

        self.nfft_box = QtWidgets.QComboBox()
        self.nfft_box.addItems(["512", "1024", "2048", "4096", "8192", "16384"])
        self.nfft_box.setCurrentText("2048")

        self.hop_box = QtWidgets.QComboBox()
        self.hop_box.addItems(["64", "128", "256", "512", "1024", "2048"])
        self.hop_box.setCurrentText("512")

        self.window_box = QtWidgets.QComboBox()
        self.window_box.addItems(["hann", "hamming", "blackman"])

        self.db_range_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.db_range_slider.setRange(20, 120)
        self.db_range_slider.setValue(80)

        self.log_freq_toggle = QtWidgets.QCheckBox("Log frequency")
        self.max_freq_spin = QtWidgets.QSpinBox()
        self.max_freq_spin.setRange(0, 48000)
        self.max_freq_spin.setValue(0)
        self.max_freq_spin.setSuffix(" Hz (0 = auto)")

        spectro_layout.addRow("FFT Size", self.nfft_box)
        spectro_layout.addRow("Hop Length", self.hop_box)
        spectro_layout.addRow("Window", self.window_box)
        spectro_layout.addRow("dB Range", self.db_range_slider)
        spectro_layout.addRow(self.log_freq_toggle)
        spectro_layout.addRow("Max Frequency", self.max_freq_spin)

        layout.addWidget(spectro_group)

        processing_group = QtWidgets.QGroupBox("Processing")
        proc_layout = QtWidgets.QVBoxLayout(processing_group)
        self.normalize_button = QtWidgets.QPushButton("Normalize")
        self.reverse_button = QtWidgets.QPushButton("Reverse")
        self.trim_button = QtWidgets.QPushButton("Trim to Selection")
        self.resample_button = QtWidgets.QPushButton("Resample")
        self.bandpass_button = QtWidgets.QPushButton("Band-pass")
        self.notch_button = QtWidgets.QPushButton("Notch")
        self.reset_button = QtWidgets.QPushButton("Reset")

        for button in (
            self.normalize_button,
            self.reverse_button,
            self.trim_button,
            self.resample_button,
            self.bandpass_button,
            self.notch_button,
            self.reset_button,
        ):
            proc_layout.addWidget(button)

        layout.addWidget(processing_group)

        decoder_group = QtWidgets.QGroupBox("Decoder Helpers")
        decoder_layout = QtWidgets.QVBoxLayout(decoder_group)
        self.sstv_button = QtWidgets.QPushButton("SSTV Tools")
        self.multimon_button = QtWidgets.QPushButton("multimon-ng")
        self.minimodem_button = QtWidgets.QPushButton("minimodem")
        decoder_layout.addWidget(self.sstv_button)
        decoder_layout.addWidget(self.multimon_button)
        decoder_layout.addWidget(self.minimodem_button)
        layout.addWidget(decoder_group)

        layout.addStretch(1)
        return panel

    def _connect_actions(self) -> None:
        self.open_action.triggered.connect(self._open_file)
        self.export_action.triggered.connect(self._export_spectrogram)
        self.autoscan_action.triggered.connect(self._run_autoscan)
        self.evidence_action.triggered.connect(self._export_evidence_bundle)
        self.undo_action.triggered.connect(self._undo)
        self.redo_action.triggered.connect(self._redo)

        self.mono_toggle.toggled.connect(self._refresh_audio)
        self.nfft_box.currentTextChanged.connect(self._refresh_spectrogram)
        self.hop_box.currentTextChanged.connect(self._refresh_spectrogram)
        self.window_box.currentTextChanged.connect(self._refresh_spectrogram)
        self.db_range_slider.valueChanged.connect(self._refresh_spectrogram)
        self.log_freq_toggle.toggled.connect(self._refresh_spectrogram)
        self.max_freq_spin.valueChanged.connect(self._refresh_spectrogram)

        self.normalize_button.clicked.connect(self._apply_normalize)
        self.reverse_button.clicked.connect(self._apply_reverse)
        self.trim_button.clicked.connect(self._apply_trim)
        self.resample_button.clicked.connect(self._apply_resample)
        self.bandpass_button.clicked.connect(self._apply_bandpass)
        self.notch_button.clicked.connect(self._apply_notch)
        self.reset_button.clicked.connect(self._reset_processing)

        self.sstv_button.clicked.connect(self._open_sstv)
        self.multimon_button.clicked.connect(self._open_multimon)
        self.minimodem_button.clicked.connect(self._open_minimodem)

        self.waveform.selection_changed.connect(self._update_selection_label)
        self.spectrogram.cursor_moved.connect(self._update_cursor_label)

    def _open_file(self) -> None:
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open audio",
            "",
            "Audio Files (*.wav *.flac *.ogg *.aiff *.aif *.mp3);;All Files (*)",
        )
        if not path_str:
            return
        try:
            self.project.load_file(Path(path_str))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
        self.file_label.setText(Path(path_str).name)
        self._refresh_audio()
        self._log(f"Loaded {path_str}")

    def _refresh_audio(self) -> None:
        self.project.mono_view = self.mono_toggle.isChecked()
        audio = self.project.get_display_audio()
        sample_rate = self.project.sample_rate or 1
        self.waveform.set_audio(audio if audio.ndim == 1 else audio[:, 0], sample_rate)
        self._refresh_spectrogram()

    def _current_spectrogram_settings(self) -> SpectrogramSettings:
        max_freq = self.max_freq_spin.value() or None
        return SpectrogramSettings(
            n_fft=int(self.nfft_box.currentText()),
            hop_length=int(self.hop_box.currentText()),
            window=self.window_box.currentText(),
            db_range=float(self.db_range_slider.value()),
            log_frequency=self.log_freq_toggle.isChecked(),
            max_frequency=max_freq,
        )

    def _refresh_spectrogram(self) -> None:
        audio = self.project.get_display_audio()
        sample_rate = self.project.sample_rate or 1
        mono_audio = audio if audio.ndim == 1 else audio[:, 0]
        self.spectrogram.update_spectrogram(mono_audio, sample_rate, self._current_spectrogram_settings())

    def _update_selection_label(self, start: float, end: float) -> None:
        self.status_bar.showMessage(f"Selection: {start:.2f}s - {end:.2f}s")

    def _update_cursor_label(self, time: float, freq: float, magnitude: float) -> None:
        self.status_bar.showMessage(f"Cursor: {time:.2f}s {freq:.1f}Hz {magnitude:.1f}dB")

    def _apply_processing(self, step: ProcessingStep) -> None:
        if self.project.working_audio is None:
            return
        self.project.apply_step(step)
        self._refresh_audio()
        self._log(f"Applied {step.name}")

    def _apply_normalize(self) -> None:
        step = ProcessingStep(
            name="Normalize",
            parameters={},
            apply=lambda audio, sr: (normalize(audio), sr),
        )
        self._apply_processing(step)

    def _apply_reverse(self) -> None:
        step = ProcessingStep(
            name="Reverse",
            parameters={},
            apply=lambda audio, sr: (reverse(audio), sr),
        )
        self._apply_processing(step)

    def _apply_trim(self) -> None:
        selection = self.waveform.selection()
        if selection is None:
            QtWidgets.QMessageBox.information(self, "Trim", "Select a region first.")
            return
        start, end = selection
        step = ProcessingStep(
            name="Trim",
            parameters={"start": start, "end": end},
            apply=lambda audio, sr: (trim(audio, sr, start, end), sr),
        )
        self._apply_processing(step)

    def _apply_resample(self) -> None:
        rate, ok = QtWidgets.QInputDialog.getInt(
            self, "Resample", "Target sample rate (Hz)", value=22050, min=8000, max=192000
        )
        if not ok:
            return
        step = ProcessingStep(
            name="Resample",
            parameters={"target_rate": rate},
            apply=lambda audio, sr: resample(audio, sr, rate),
        )
        self._apply_processing(step)

    def _apply_bandpass(self) -> None:
        low, ok = QtWidgets.QInputDialog.getDouble(self, "Band-pass", "Low Hz", value=300.0, min=0.0)
        if not ok:
            return
        high, ok = QtWidgets.QInputDialog.getDouble(self, "Band-pass", "High Hz", value=3000.0, min=0.0)
        if not ok:
            return
        order, ok = QtWidgets.QInputDialog.getInt(self, "Band-pass", "Order", value=4, min=1, max=12)
        if not ok:
            return
        settings = FilterSettings(low_hz=low, high_hz=high, order=order)
        step = ProcessingStep(
            name="Band-pass",
            parameters=asdict(settings),
            apply=lambda audio, sr: (bandpass(audio, sr, settings), sr),
        )
        self._apply_processing(step)

    def _apply_notch(self) -> None:
        center, ok = QtWidgets.QInputDialog.getDouble(self, "Notch", "Center Hz", value=1000.0, min=0.0)
        if not ok:
            return
        q, ok = QtWidgets.QInputDialog.getDouble(self, "Notch", "Q", value=30.0, min=1.0)
        if not ok:
            return
        settings = NotchSettings(center_hz=center, q=q)
        step = ProcessingStep(
            name="Notch",
            parameters=asdict(settings),
            apply=lambda audio, sr: (notch(audio, sr, settings), sr),
        )
        self._apply_processing(step)

    def _reset_processing(self) -> None:
        self.project.reset()
        self._refresh_audio()
        self._log("Reset to original audio")

    def _undo(self) -> None:
        self.project.undo()
        self._refresh_audio()
        self._log("Undo")

    def _redo(self) -> None:
        self.project.redo()
        self._refresh_audio()
        self._log("Redo")

    def _export_spectrogram(self) -> None:
        if self.spectrogram.result is None:
            return
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Spectrogram", "spectrogram.png", "PNG Image (*.png)"
        )
        if not path_str:
            return
        selection = self.waveform.selection()
        if selection is not None and QtWidgets.QMessageBox.question(
            self, "Export", "Export selection only?"
        ) == QtWidgets.QMessageBox.StandardButton.Yes:
            min_freq, ok = QtWidgets.QInputDialog.getDouble(
                self, "Export", "Min frequency (Hz, 0 = full)", value=0.0, min=0.0
            )
            if not ok:
                return
            max_freq, ok = QtWidgets.QInputDialog.getDouble(
                self, "Export", "Max frequency (Hz, 0 = full)", value=0.0, min=0.0
            )
            if not ok:
                return
            self._export_selection_image(Path(path_str), selection, min_freq or None, max_freq or None)
        else:
            exporter = pg.exporters.ImageExporter(self.spectrogram.plot_item)
            exporter.export(path_str)
        metadata_path = Path(path_str).with_suffix(".json")
        metadata = {
            "settings": self._current_spectrogram_settings().__dict__,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._log(f"Exported spectrogram to {path_str}")

    def _run_autoscan(self) -> None:
        if self.project.working_audio is None or self.project.sample_rate is None:
            return
        output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select autoscan output folder")
        if not output_dir:
            return
        audio = self.project.get_display_audio()
        mono_audio = audio if audio.ndim == 1 else audio[:, 0]
        selection = self.waveform.selection()
        if selection is not None:
            start, end = selection
            mono_audio = trim(mono_audio, self.project.sample_rate, start, end)
        self._log("Autoscan started...")
        self.autoscan_worker = AutoscanWorker(mono_audio, self.project.sample_rate, Path(output_dir))
        self.autoscan_worker.finished_with_manifest.connect(self._autoscan_finished)
        self.autoscan_worker.failed.connect(self._autoscan_failed)
        self.autoscan_worker.start()

    def _autoscan_finished(self, manifest: object) -> None:
        self._log("Autoscan finished.")
        if hasattr(manifest, "output_dir"):
            self._log(f"Autoscan output: {manifest.output_dir}")

    def _autoscan_failed(self, message: str) -> None:
        self._log(f"Autoscan failed: {message}")
        QtWidgets.QMessageBox.critical(self, "Autoscan error", message)

    def _open_sstv(self) -> None:
        if not self.sstv_plugin.is_available():
            QtWidgets.QMessageBox.information(
                self,
                "SSTV",
                "Install qsstv or sstv to enable SSTV decoding.",
            )
            return
        if "Launch QSSTV" in self.sstv_plugin.available_modes():
            self.sstv_plugin.launch_qsstv()
            self._log("Launching QSSTV...")
        worker = self.sstv_plugin.decode_with_sstv(
            self._export_temp_wav(), self._decoder_output_dir("sstv")
        )
        if worker is not None:
            self._run_decoder_tool("sstv", worker)

    def _open_multimon(self) -> None:
        if not self.multimon_plugin.is_available():
            QtWidgets.QMessageBox.information(
                self,
                "multimon-ng",
                "Install multimon-ng to enable decoding.",
            )
            return
        self._run_decoder_tool(
            tool_name="multimon-ng",
            command_worker=self.multimon_plugin.decode(
                self._export_temp_wav(),
                self._decoder_output_dir("multimon"),
                ["DTMF", "AFSK1200", "AFSK2400", "FSK9600"],
            ),
        )

    def _open_minimodem(self) -> None:
        if not self.minimodem_plugin.is_available():
            QtWidgets.QMessageBox.information(
                self,
                "minimodem",
                "Install minimodem to enable decoding.",
            )
            return
        args, ok = QtWidgets.QInputDialog.getText(
            self,
            "minimodem",
            "Arguments (e.g. --rtty 45.45)",
            text="--rtty 45.45",
        )
        if not ok:
            return
        self._run_decoder_tool(
            tool_name="minimodem",
            command_worker=self.minimodem_plugin.decode(
                self._export_temp_wav(),
                self._decoder_output_dir("minimodem"),
                args.split(),
            ),
        )

    def _run_decoder_tool(self, tool_name: str, command_worker: QtCore.QThread | None) -> None:
        if command_worker is None:
            QtWidgets.QMessageBox.warning(self, tool_name, "Tool is not available.")
            return
        command_worker.finished_with_result.connect(self._decoder_finished)
        command_worker.start()
        self._log(f"Running {tool_name}...")

    def _decoder_finished(self, result: object) -> None:
        self._log(f"Command: {result.command}")
        self._log(f"Return code: {result.return_code}")
        if result.stdout:
            self._log(result.stdout)
        if result.stderr:
            self._log(result.stderr)

    def _export_temp_wav(self) -> Path:
        if self.project.working_audio is None or self.project.sample_rate is None:
            raise RuntimeError("No audio loaded.")
        output_dir = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.TempLocation))
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_path = output_dir / "argaudioworkbench_temp.wav"
        import soundfile as sf

        sf.write(str(wav_path), self.project.get_display_audio(), self.project.sample_rate)
        return wav_path

    def _decoder_output_dir(self, name: str) -> Path:
        base = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation))
        return base / "ArgAudioWorkbench" / "decoder_logs" / name

    def _log(self, message: str) -> None:
        self.log_panel.appendPlainText(message)

    def _export_evidence_bundle(self) -> None:
        if self.project.working_audio is None or self.project.sample_rate is None:
            return
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Evidence Bundle", "evidence.zip", "Zip Archive (*.zip)"
        )
        if not path_str:
            return
        temp_image = self._export_temp_image()
        metadata = EvidenceMetadata(
            audio_path=str(self.project.audio_path) if self.project.audio_path else "",
            audio_sha256=hash_audio(self.project.working_audio),
            processing_steps=[{"name": step.name, "parameters": step.parameters} for step in self.project.history],
            spectrogram_settings=self._current_spectrogram_settings().__dict__,
        )
        image_paths = [temp_image] if temp_image else []
        save_evidence_bundle(Path(path_str), metadata, image_paths)
        self._log(f"Saved evidence bundle to {path_str}")

    def _export_selection_image(
        self,
        output_path: Path,
        selection: tuple[float, float],
        min_freq: float | None,
        max_freq: float | None,
    ) -> None:
        if self.spectrogram.result is None:
            return
        start, end = selection
        result = self.spectrogram.result
        time_mask = (result.times >= start) & (result.times <= end)
        freq_mask = np.ones(result.frequencies.shape, dtype=bool)
        if min_freq is not None:
            freq_mask &= result.frequencies >= min_freq
        if max_freq is not None:
            freq_mask &= result.frequencies <= max_freq
        cropped = result.magnitude_db[freq_mask][:, time_mask]
        freqs = result.frequencies[freq_mask]
        if cropped.size == 0:
            return
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        extent = [start, end, freqs.min(), freqs.max()]
        vmin = cropped.max() - self.spectrogram.settings.db_range
        vmax = cropped.max()
        ax.imshow(cropped, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _export_temp_image(self) -> Path | None:
        if self.spectrogram.result is None:
            return None
        output_dir = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.TempLocation))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "argaudioworkbench_spectrogram.png"
        exporter = pg.exporters.ImageExporter(self.spectrogram.plot_item)
        exporter.export(str(path))
        return path
