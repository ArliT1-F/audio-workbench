from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


class WaveformWidget(QtWidgets.QWidget):
    selection_changed = QtCore.Signal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#151515")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.setLabel("bottom", "Time (s)")
        layout.addWidget(self.plot_widget)

        self.plot_item = self.plot_widget.plot(pen=pg.mkPen("#4FC3F7"))
        self.region = pg.LinearRegionItem(values=(0, 1), brush=pg.mkBrush(50, 50, 50, 50))
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.plot_widget.addItem(self.region)
        self.region.setVisible(False)

        self._audio = np.array([], dtype=np.float32)
        self._sample_rate = 1

    def set_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        self._audio = audio
        self._sample_rate = sample_rate
        if audio.size == 0:
            self.plot_item.setData([])
            self.region.setVisible(False)
            return
        times = np.arange(audio.size) / sample_rate
        self.plot_item.setData(times, audio)
        self.region.setRegion((times[0], times[-1]))
        self.region.setVisible(True)

    def selection(self) -> tuple[float, float] | None:
        if not self.region.isVisible():
            return None
        start, end = self.region.getRegion()
        return float(start), float(end)

    def _on_region_changed(self) -> None:
        selection = self.selection()
        if selection is None:
            return
        self.selection_changed.emit(*selection)
