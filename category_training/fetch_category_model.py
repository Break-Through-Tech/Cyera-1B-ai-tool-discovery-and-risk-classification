"""Download the trained Kaggle-14 category model instead of retraining it.

Fetches the GitHub Release asset, checks its SHA-256, extracts it to
models/categories/kaggle14_run1, and verifies every file against the run's own
manifest. Standard library only. Refuses to overwrite an existing model folder.

    python category_training/fetch_category_model.py

Then categorize.py / run_pipeline.py can use --model-dir models/categories/kaggle14_run1.
Only load joblib files from a source you trust; this script checks the published hash.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Break-Through-Tech/Cyera-1B-ai-tool-discovery-and-risk-classification"
TAG = "category-model-kaggle14-run1"
ASSET = "category-model-kaggle14_run1.zip"
SHA256 = "3d8587350a663ac7eb5108ffb5afceeae4f133070fc5a1ca9a7d12eb82603351"
DEST = ROOT / "models/categories"
RUN = "kaggle14_run1"


def main() -> int:
    target = DEST / RUN
    if target.exists():
        print(f"{target.relative_to(ROOT)} already exists; remove it first to re-download.")
        return 1
    url = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"SHA-256 mismatch: got {digest}, expected {SHA256}. Nothing extracted.")
        return 1
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        if any(not n.startswith(RUN + "/") or ".." in Path(n).parts for n in names):
            print("Unexpected paths in archive. Nothing extracted.")
            return 1
        DEST.mkdir(parents=True, exist_ok=True)
        z.extractall(DEST)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, h in manifest["artifacts"].items()
           if hashlib.sha256((target / n).read_bytes()).hexdigest() != h]
    if bad:
        print(f"Extracted files do not match the run manifest: {bad}")
        return 1
    print(f"OK: {len(manifest['artifacts'])} files verified in {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
