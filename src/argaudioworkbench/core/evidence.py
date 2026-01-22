from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EvidenceMetadata:
    audio_path: str
    audio_sha256: str
    processing_steps: list[dict]
    spectrogram_settings: dict


def hash_audio(audio: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(audio.tobytes())
    return digest.hexdigest()


def save_evidence_bundle(
    output_zip: Path,
    metadata: EvidenceMetadata,
    image_paths: list[Path],
) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata.__dict__, indent=2))
        for image_path in image_paths:
            archive.write(image_path, arcname=f"images/{image_path.name}")
