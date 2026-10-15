"""Raw-data provenance: checksums, immutability, manifest."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def build_manifest(raw_dir: Path, source: dict) -> dict:
    files = []
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.is_file() and p.name != ".gitkeep":
            files.append({"path": str(p.relative_to(raw_dir)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "source": source,
            "n_files": len(files), "files": files}


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify_manifest(raw_dir: Path, manifest_path: Path) -> list[str]:
    """Return a list of problems; empty means every raw file matches its checksum."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    problems = []
    for f in m["files"]:
        p = Path(raw_dir) / f["path"]
        if not p.exists():
            problems.append(f"missing: {f['path']}")
        elif sha256(p) != f["sha256"]:
            problems.append(f"checksum mismatch (raw file modified): {f['path']}")
    return problems


def make_read_only(raw_dir: Path) -> None:
    for p in Path(raw_dir).rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
