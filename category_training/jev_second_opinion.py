"""Ask Jev for a second-opinion category for every AI tool in the dashboard runs.

Uses exactly the setup measured in docs/jev_category_evaluation.md: same question,
same category definitions (built from development rows), same description-only
text contract. One call per distinct application endpoint across all companies in
dashboard/companies.json; responses are cached, so re-runs are free.

Description source per app:
  * catalog-matched apps: the catalog's product description (by tool_id)
  * researched apps: the "Product Description" section of the saved web research
    report, extracted with the pipeline's own extract_report()

Output: data/processed/dashboard_runs/jev_second_opinion.json (read by the dashboard).
Never changes the assigned category; this is advisory evidence only.

    python category_training/jev_second_opinion.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "category_training"))
sys.path.insert(0, str(ROOT / "preprocessing_scripts"))

from jev_category_eval import (  # noqa: E402
    INSTRUCTIONS, VERSION, build_criteria, call_jev, load_rows, slug, write_json,
)
from prepare_category_data import build_model_text, clean_text, extract_report  # noqa: E402

MODEL_DIR = ROOT / "models/categories/kaggle14_run1"
PREPARED = ROOT / "data/processed/category/category_preparation_v1"
CATALOG = ROOT / "data/processed/kaggle_dataset_b/kaggle_dataset_b_filled.csv"
CACHE = ROOT / "data/cache/jev_category_cache.json"
OUT = ROOT / "data/processed/dashboard_runs/jev_second_opinion.json"
JEV_MODEL = "jev-latest"


def description_for(record: dict, catalog: dict) -> tuple[str, str]:
    for report in record.get("reports") or []:
        text = extract_report(report).get("description_only_text") or ""
        if text.strip():
            return text, "web research report (Product Description section)"
    for rec in (record.get("categorization") or {}).get("catalog_records") or []:
        desc = catalog.get(rec.get("tool_id"), "")
        if desc.strip():
            return build_model_text(clean_text(desc), mode="description_only"), "catalog description"
    return "", "none"


def main():
    companies = json.loads((ROOT / "dashboard/companies.json").read_text(encoding="utf-8"))["companies"]
    taxonomy = json.loads((MODEL_DIR / "taxonomy.json").read_text(encoding="utf-8"))["categories"]
    by_slug = {slug(c): c for c in taxonomy}
    dev, _ = load_rows(PREPARED, MODEL_DIR / "split_manifest.csv")
    criteria = build_criteria(dev, taxonomy)
    with open(CATALOG, newline="", encoding="utf-8-sig") as f:
        catalog = {r["tool_id"]: r.get("description", "") for r in csv.DictReader(f)}

    apps = {}
    for entry in companies:
        report = json.loads((ROOT / entry["run_dir"] / "categorized_applications.json").read_text(encoding="utf-8"))
        for rec in report["results"]:
            if rec.get("is_ai_tool") is not True:
                continue
            endpoint = rec.get("endpoint") or rec["observed_identity"]["observed_endpoint"]
            if endpoint in apps:
                continue
            text, source = description_for(rec, catalog)
            apps[endpoint] = {"name": rec["application_name"], "text": text, "evidence_source": source}

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {"version": VERSION, "entries": {}}

    def request_key(text):   # identical to jev_category_eval.request_key
        spec = {"v": VERSION, "model": JEV_MODEL, "instructions": INSTRUCTIONS,
                "criteria": criteria, "evidence": text}
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    todo = [e for e, a in apps.items() if a["text"] and request_key(a["text"]) not in cache["entries"]]
    print(f"AI apps: {len(apps)}; cached: {len(apps) - len(todo)}; new Jev calls: {len(todo)}")
    if todo:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("Set TYPESAFE_API_KEY in the project-root .env. No API calls were made.")
        for endpoint in todo:
            text = apps[endpoint]["text"]
            result = call_jev(JEV_MODEL, text, criteria, 60, key)
            cache["entries"][request_key(text)] = {"endpoint": endpoint, "result": result,
                                                   "retrieved_at": datetime.now(timezone.utc).isoformat()}
            write_json(CACHE, cache)
            print(f"  {apps[endpoint]['name']}: {by_slug[result['choice']]} ({result['probabilities'][result['choice']]:.2f})")

    out = {}
    for endpoint, a in apps.items():
        entry = cache["entries"].get(request_key(a["text"])) if a["text"] else None
        if not entry:
            out[endpoint] = {"name": a["name"], "status": "no_description", "evidence_source": a["evidence_source"]}
            continue
        res = entry["result"]
        top = sorted(res["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        out[endpoint] = {"name": a["name"], "status": "ok", "category": by_slug[res["choice"]],
                         "probability": round(res["probabilities"][res["choice"]], 3),
                         "top_three": [{"category": by_slug[k], "probability": round(v, 3)} for k, v in top],
                         "evidence_source": a["evidence_source"], "returned_model": res.get("returned_model")}
    write_json(OUT, {
        "version": "jev-second-opinion-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "jev_model_requested": JEV_MODEL,
        "criteria_sha256": hashlib.sha256(json.dumps(criteria, sort_keys=True).encode()).hexdigest(),
        "evaluation": "docs/jev_category_evaluation.md",
        "note": "Advisory second opinion; never overrides the assigned category. Probabilities are provider outputs.",
        "apps": out})
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
