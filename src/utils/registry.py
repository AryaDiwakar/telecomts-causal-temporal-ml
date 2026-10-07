"""Append experiment results to outputs/results_registry.json (newest last)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "outputs" / "results_registry.json"


def load() -> list[dict]:
    if REGISTRY.exists():
        with open(REGISTRY) as fh:
            return json.load(fh)
    return []


def append(record: dict) -> None:
    records = load()
    records.append(record)
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"registry: appended {record.get('experiment', '?')} -> {REGISTRY}")


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
