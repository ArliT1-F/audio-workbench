# ArgAudioWorkbench

ArgAudioWorkbench is a cross-platform (Linux-first) audio analysis workstation for ARG solving. It focuses on quickly revealing spectrogram art (hidden images, glyphs, or words etched into audio) and providing an integrated workflow for decoding signals like SSTV, DTMF, and RTTY/FSK.

## Features

- **Spectrogram art discovery** with fast parameter iteration (FFT size, hop, window, dB range).
- **Autoscan**: one-click export of a grid of spectrograms with multiple parameter combinations.
- **Waveform + selection** with zoom/pan and cursor readout.
- **Non-destructive processing** with undo/redo: normalize, reverse, trim, resample, band-pass, notch.
- **Decoder helpers** (plugin-based) for SSTV, multimon-ng, and minimodem (external tools if installed).
- **Evidence capture**: export spectrograms and bundle metadata + hashes.

## Install (Linux Mint recommended)

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libsndfile1 ffmpeg
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

> **Optional decoders** (install via your distro package manager):
>
>- **QSSTV / sstv**: `sudo apt install qsstv` or `sudo apt install sstv` (varies by distro)
>- **multimon-ng**: `sudo apt install multimon-ng`
>- **minimodem**: `sudo apt install minimodem`

> **Optional MP3 support**: install `ffmpeg` and then `pip install -e .[mp3]`.

## Windows/macOS (best effort)

- Install Python 3.11+.
- Install dependencies via `pip install -e .`.
- For MP3 support, install FFmpeg and ensure it is on PATH.
- Some external decoders may not be available on all platforms.

## Quickstart: Reveal etched images in audio

1. Open an audio file (`Ctrl+O`).
2. Switch to **Spectrogram** view.
3. Try **Autoscan** to generate a grid of parameterized spectrograms or tweak FFT size/hop/window/dB range.
4. Export the best-looking spectrogram (`Ctrl+E`).
5. If needed, apply a filter (band-pass/notch) and re-run Autoscan.

## Developer scripts

- `scripts/run_dev.sh`: creates a venv and runs the app.
- `scripts/build_wheel.sh`: builds a wheel using hatchling.

## License

MIT. See [LICENSE](LICENSE).
