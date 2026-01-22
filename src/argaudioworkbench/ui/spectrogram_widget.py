from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from argaudioworkbench.dsp.stft import SpectrogramResult, SpectrogramSettings, render_spectrogram


class SpectrogramWidget(QtWidgets.QWidget):
    cursor_moved = QtCore.Signal(float, float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground("#111")
        self.plot_item = self.view.addPlot()
        self.plot_item.setLabel("left", "Frequency (Hz)")
        self.plot_item.setLabel("bottom", "Time (s)")
        self.plot_item.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.view)

        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        self.color_map = pg.colormap.get("inferno")
        self.image_item.setLookupTable(self.color_map.getLookupTable())

        self._settings = SpectrogramSettings()
        self._result: SpectrogramResult | None = None
        self._cursor_proxy = pg.SignalProxy(self.plot_item.scene().sigMouseMoved, rateLimit=30, slot=self._on_mouse_move)

    def update_spectrogram(self, audio: np.ndarray, sample_rate: int, settings: SpectrogramSettings) -> None:
        if audio.size == 0:
            self.image_item.setImage(np.zeros((1, 1)))
            return
        self._settings = settings
        self._result = render_spectrogram(audio, sample_rate, settings)
        vmin = self._result.magnitude_db.max() - settings.db_range
        vmax = self._result.magnitude_db.max()
        self.image_item.setImage(self._result.magnitude_db, levels=(vmin, vmax))
        self.image_item.resetTransform()
        self.image_item.setRect(
            QtCore.QRectF(
                self._result.times.min(),
                self._result.frequencies.min(),
                self._result.times.max() - self._result.times.min(),
                self._result.frequencies.max() - self._result.frequencies.min(),
            )
        )

    def _on_mouse_move(self, evt: tuple) -> None:
        if self._result is None:
            return
        pos = evt[0]
        if not self.plot_item.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_item.vb.mapSceneToView(pos)
        time = float(mouse_point.x())
        freq = float(mouse_point.y())
        magnitude = self._lookup_magnitude(time, freq)
        self.cursor_moved.emit(time, freq, magnitude)

    def _lookup_magnitude(self, time: float, freq: float) -> float:
        if self._result is None:
            return 0.0
        time_idx = np.searchsorted(self._result.times, time)
        freq_idx = np.searchsorted(self._result.frequencies, freq)
        time_idx = np.clip(time_idx, 0, self._result.times.size - 1)
        freq_idx = np.clip(freq_idx, 0, self._result.frequencies.size - 1)
        return float(self._result.magnitude_db[freq_idx, time_idx])

    @property
    def settings(self) -> SpectrogramSettings:
        return self._settings

    @property
    def result(self) -> SpectrogramResult | None:
        return self._result
