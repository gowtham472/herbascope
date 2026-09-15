"""Download and pin every external asset the pipeline needs, so later steps run offline.

Assets (all pinned in ml/configs/pipeline.json):
  * Mikrobat microscopy dataset  -> data/raw/mikrobat/   (GitHub archive at a fixed commit)
  * DIMPSAR far-OOD source       -> data/raw/dimpsar/    (Hugging Face parquet at a fixed revision)
  * DINOv2 backbone weights      -> models/pretrained/<backbone>/  (pinned hub revision)

Each download writes a SOURCE.json receipt (URL, revision, SHA-256, timestamp). Existing
assets whose receipt matches the pinned revision are left untouched; data/raw is immutable.

Usage:  python -m scripts.download_assets
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from ml import paths
from ml.encoders.dinov2_encoder import ensure_backbone
from ml.pipeline_config import load_pipeline_config

CHUNK_BYTES = 1 << 20


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "herbascope-x-pipeline"})
    buffer = io.BytesIO()
    with urllib.request.urlopen(request, timeout=120) as response:
        while chunk := response.read(CHUNK_BYTES):
            buffer.write(chunk)
    return buffer.getvalue()


def _receipt_matches(directory: Path, key: str, expected: str) -> bool:
    receipt = directory / "SOURCE.json"
    if not receipt.is_file():
        return False
    return json.loads(receipt.read_text(encoding="utf-8")).get(key) == expected


def _write_receipt(directory: Path, receipt: dict) -> None:
    (directory / "SOURCE.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def download_mikrobat() -> None:
    source = load_pipeline_config().sources.mikrobat
    target = paths.RAW_DIR / "mikrobat"
    if _receipt_matches(target, "commit", source.commit):
        print(f"[mikrobat] already present at commit {source.commit[:7]}")
        return
    print(f"[mikrobat] downloading {source.archive_url}")
    archive = _fetch(source.archive_url)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for member in bundle.infolist():
            # GitHub archives nest everything under "<repo>-<commit>/"; strip that prefix.
            relative = Path(*Path(member.filename).parts[1:])
            if member.is_dir() or not relative.parts:
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bundle.read(member))
    _write_receipt(
        target,
        {
            "dataset": source.name,
            "repository": source.repository,
            "commit": source.commit,
            "archive_url": source.archive_url,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "downloaded_at": _utc_now(),
        },
    )
    print(f"[mikrobat] extracted to {target}")


def download_dimpsar() -> None:
    source = load_pipeline_config().sources.dimpsar
    target = paths.RAW_DIR / "dimpsar"
    if _receipt_matches(target, "sha256", source.sha256):
        print(f"[dimpsar] already present (revision {source.revision[:7]})")
        return
    print(f"[dimpsar] downloading {source.file_url}")
    payload = _fetch(source.file_url)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source.sha256:
        raise RuntimeError(f"DIMPSAR checksum mismatch: expected {source.sha256}, got {digest}")
    target.mkdir(parents=True, exist_ok=True)
    (target / Path(source.file).name).write_bytes(payload)
    _write_receipt(
        target,
        {
            "dataset": source.name,
            "repository": source.repository,
            "revision": source.revision,
            "file": source.file,
            "sha256": digest,
            "license": source.license,
            "downloaded_at": _utc_now(),
        },
    )
    print(f"[dimpsar] saved to {target}")


def main() -> int:
    download_mikrobat()
    download_dimpsar()
    backbone = load_pipeline_config().encoder.backbone
    downloaded = ensure_backbone(
        backbone.name,
        backbone.hub_id,
        backbone.revision,
        backbone.license,
        paths.PRETRAINED_DIR / backbone.local_dir,
    )
    print(f"[encoder] {backbone.name} {'downloaded' if downloaded else 'already cached'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
