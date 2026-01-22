from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

from argaudioworkbench.dsp.stft import SpectrogramSettings, render_spectrogram

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class AutoscanEntry:
    settings: SpectrogramSettings
    image_path: Path


@dataclass(frozen=True)
class AutoscanManifest:
    entries: list[AutoscanEntry]
    output_dir: Path

    def to_json(self) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "entries": [
                {
                    "settings": {
                        "n_fft": entry.settings.n_fft,
                        "hop_length": entry.settings.hop_length,
                        "window": entry.settings.window,
                        "db_range": entry.settings.db_range,
                        "log_frequency": entry.settings.log_frequency,
                        "max_frequency": entry.settings.max_frequency,
                    },
                    "image_path": str(entry.image_path),
                }
                for entry in self.entries
            ],
        }


def autoscan_parameters() -> Iterable[SpectrogramSettings]:
    for n_fft in (1024, 2048, 4096, 8192):
        for hop_length in (n_fft // 8, n_fft // 4):
            for db_range in (50.0, 70.0, 90.0):
                for log_frequency in (False, True):
                    yield SpectrogramSettings(
                        n_fft=n_fft,
                        hop_length=hop_length,
                        window="hann",
                        db_range=db_range,
                        log_frequency=log_frequency,
                    )


def run_autoscan(
    audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
) -> AutoscanManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[AutoscanEntry] = []

    for settings in autoscan_parameters():
        result = render_spectrogram(audio, sample_rate, settings)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        extent = [result.times.min(), result.times.max(), result.frequencies.min(), result.frequencies.max()]
        vmin = result.magnitude_db.max() - settings.db_range
        vmax = result.magnitude_db.max()
        ax.imshow(result.magnitude_db, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="inferno")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(
            f"nfft={settings.n_fft} hop={settings.hop_length} db={settings.db_range} log={settings.log_frequency}"
        )
        filename = (
            f"spec_nfft-{settings.n_fft}_hop-{settings.hop_length}_db-{int(settings.db_range)}_log-"
            f"{int(settings.log_frequency)}.png"
        )
        image_path = output_dir / filename
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)
        entries.append(AutoscanEntry(settings=settings, image_path=image_path))

    manifest = AutoscanManifest(entries=entries, output_dir=output_dir)
    manifest_path = output_dir / "autoscan_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_json(), handle, indent=2)

    _write_gallery(output_dir, entries)

    return manifest


def _write_gallery(output_dir: Path, entries: list[AutoscanEntry]) -> None:
    html_lines = [
        "<html><head><title>ArgAudioWorkbench Autoscan</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee} .grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:12px;} img{width:100%;border:1px solid #333}</style>",
        "</head><body>",
        "<h1>Autoscan Results</h1>",
        "<div class=\"grid\">",
    ]
    for entry in entries:
        label = (
            f"nfft={entry.settings.n_fft} hop={entry.settings.hop_length} db={entry.settings.db_range} log={entry.settings.log_frequency}"
        )
        html_lines.append(
            f"<div><img src=\"{entry.image_path.name}\" alt=\"{label}\"><p>{label}</p></div>"
        )
    html_lines.extend(["</div>", "</body></html>"])
    (output_dir / "index.html").write_text("\n".join(html_lines), encoding="utf-8")
