import numpy as np

from argaudioworkbench.core.autoscan import autoscan_parameters, run_autoscan


def test_autoscan_manifest(tmp_path) -> None:
    sample_rate = 8000
    audio = np.random.randn(sample_rate).astype(np.float32)
    manifest = run_autoscan(audio, sample_rate, tmp_path)
    assert manifest.output_dir == tmp_path
    assert len(manifest.entries) == len(list(autoscan_parameters()))
    assert (tmp_path / "autoscan_manifest.json").exists()
    assert (tmp_path / "index.html").exists()
