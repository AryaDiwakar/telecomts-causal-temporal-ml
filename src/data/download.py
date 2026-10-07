"""Download TelecomTS from the Hugging Face Hub into data/raw/.

Usage:
    python -m src.data.download
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "TelecomTS"
REPO_ID = "AliMaatouk/TelecomTS"


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from exc

    print(f"Downloading {REPO_ID} -> {DEST}")
    path = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=str(DEST),
    )
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
