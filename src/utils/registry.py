"""Append experiment results to outputs/results_registry.json (newest last).

Phase 6 (docs/benchmark_audit_v3.md) upgrade: every new record is enriched with
provenance so a run is only ``COMPLETE`` when its dataset/split/code identity is
recorded. Old records without these fields remain valid and readable.

Provenance added on append:
  * ``code_revision``  : short git hash (``-dirty`` if the tree is dirty)
  * ``config_hash``    : sha256 of the config files used
  * ``data_fingerprint``: sha256 of split_indices/targets/scaler/rca_templates
  * ``status``         : PLANNED | RUNNING | COMPLETE | FAILED | INVALID | RESULT_PENDING
  * ``timestamp``      : ISO-8601 UTC
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "outputs" / "results_registry.json"
PROCESSED = ROOT / "data" / "processed"

VALID_STATUS = {"PLANNED", "RUNNING", "COMPLETE", "FAILED", "INVALID", "RESULT_PENDING"}

# files whose content defines a result's data identity
_DATA_FILES = ["split_indices.npz", "targets.npz", "scaler.json", "rca_templates.json"]
_CONFIG_FILES = [
    "configs/data.yaml", "configs/model.yaml", "configs/training.yaml",
    "configs/experiments.yaml", "configs/pretrain.yaml", "configs/synthetic.yaml",
]


def _sha256_of_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        if p.exists():
            h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def code_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip()
        if not rev:
            return "unknown"
        return f"{rev}-dirty" if dirty else rev
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def data_fingerprint() -> str:
    return _sha256_of_files([PROCESSED / f for f in _DATA_FILES])


def config_hash() -> str:
    return _sha256_of_files([ROOT / f for f in _CONFIG_FILES])


def enrich(record: dict) -> dict:
    """Return ``record`` with provenance fields filled in (without clobbering)."""
    out = dict(record)
    out.setdefault("code_revision", code_revision())
    out.setdefault("config_hash", config_hash())
    out.setdefault("data_fingerprint", data_fingerprint())
    out.setdefault("status", "COMPLETE")
    out.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return out


def load() -> list[dict]:
    if REGISTRY.exists():
        with open(REGISTRY) as fh:
            return json.load(fh)
    return []


def append(record: dict) -> None:
    status = record.get("status", "COMPLETE")
    if status not in VALID_STATUS:
        raise ValueError(f"invalid registry status {status!r}; allowed: {sorted(VALID_STATUS)}")
    records = load()
    records.append(enrich(record))
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"registry: appended {record.get('experiment', '?')} ({status}) -> {REGISTRY}")


def deduplicate() -> None:
    """Keep the newest row for each reproducible experiment identity."""
    records = load()
    newest: dict[tuple, dict] = {}
    order: list[tuple] = []
    for record in records:
        key = (
            record.get("experiment"),
            record.get("seed"),
            record.get("task"),
            record.get("model"),
            record.get("variant"),
        )
        if key not in newest:
            order.append(key)
        newest[key] = record
    with open(REGISTRY, "w") as fh:
        json.dump([newest[key] for key in order], fh, indent=2)
    print(f"registry: deduplicated {len(records)} -> {len(order)} rows")


def update(experiment: str, fields: dict) -> None:
    records = load()
    for r in records:
        if r.get("experiment") == experiment and r.get("seed") == fields.get("seed"):
            r.update(fields)
    with open(REGISTRY, "w") as fh:
        json.dump(records, fh, indent=2)


def audit() -> dict:
    """Summarise provenance coverage; used by tests and the report."""
    records = load()
    n = len(records)
    have_rev = sum(bool(r.get("code_revision")) for r in records)
    have_fp = sum(bool(r.get("data_fingerprint")) for r in records)
    have_status = sum(bool(r.get("status")) for r in records)
    fingerprints: dict[str, int] = {}
    for r in records:
        fp = r.get("data_fingerprint", "MISSING")
        fingerprints[fp] = fingerprints.get(fp, 0) + 1
    return {
        "n": n,
        "with_code_revision": have_rev,
        "with_data_fingerprint": have_fp,
        "with_status": have_status,
        "distinct_data_fingerprints": len(fingerprints),
        "fingerprint_counts": fingerprints,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
