"""Build the trusted (human-labelled) test set: items, a blind labelling sheet, and
model predictions stored separately.

Selection (seed 42, reproducible):
  * every application in the Summit Financial usage log (covers NovaTech's too)
  * 8 Kaggle AI tools per category from the category model's held-out TEST split,
    excluding the 154 tools already used in docs/jev_category_evaluation.md
  * 16 conventional (non-AI) enterprise tools from the finalized reference set

Outputs in evaluation/trusted_test_set/:
  labeling_sheet.xlsx   what labellers fill in: name, website, description ONLY
  items.csv             key: source, source labels, text (do not show labellers)
  predictions.csv       local TF-IDF and Jev predictions on the same text (hidden)

    python evaluation/build_trusted_test_set.py            # needs TYPESAFE_API_KEY for new Jev answers
    python evaluation/build_trusted_test_set.py --no-jev   # local model only

Labellers must not open items.csv or predictions.csv before labelling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "category_training"))
sys.path.insert(0, str(ROOT / "preprocessing_scripts"))
from jev_category_eval import INSTRUCTIONS, VERSION, call_jev, slug, write_json  # noqa: E402
from prepare_category_data import build_model_text, extract_report  # noqa: E402

OUT = ROOT / "evaluation/trusted_test_set"
MODEL_DIR = ROOT / "models/categories/kaggle14_run1"
PREPARED = ROOT / "data/processed/category/category_preparation_v1"
USAGE_RUN = ROOT / "data/processed/dashboard_runs/summit_reviewed/categorized_applications.json"
CATALOG = ROOT / "data/processed/kaggle_dataset_b/kaggle_dataset_b_filled.csv"
REFERENCE = ROOT / "data/processed/final/dataset_b_final_kaggle_categories.csv"
JEV_EVAL = ROOT / "artifacts/jev_category_eval/predictions.csv"
CRITERIA = ROOT / "artifacts/jev_category_eval/criteria.json"
JEV_CACHE = ROOT / "data/cache/jev_category_cache.json"
PER_CATEGORY, NON_AI, SEED = 8, 16, 42


def rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def norm(name):
    return " ".join(name.lower().split())


def usage_items(catalog, reference):
    """Every app in the usage log, with the best available product description."""
    report = json.loads(USAGE_RUN.read_text(encoding="utf-8"))
    by_name = {norm(r["tool_name"]): r for r in reference if r.get("description", "").strip()}
    items = []
    for rec in report["results"]:
        name = json.loads(rec["observed_identity"]["name_variants"])[0]
        text, source = "", ""
        for rep in rec.get("reports") or []:
            text = extract_report(rep).get("description_only_text") or ""
            if text.strip():
                source = "web research description"
                break
        if not text.strip():
            for c in (rec.get("categorization") or {}).get("catalog_records") or []:
                desc = catalog.get(c.get("tool_id"), "")
                if desc.strip():
                    text, source = build_model_text(desc, mode="description_only"), "catalog description"
                    break
        if not text.strip() and norm(name) in by_name:
            text = build_model_text(by_name[norm(name)]["description"], mode="description_only")
            source = "reference description"
        cat = (rec.get("categorization") or {})
        items.append({"source": "usage_log", "source_id": rec["application_id"], "tool_name": name,
                      "website": rec.get("endpoint") or "", "text": text, "text_source": source,
                      "source_is_ai": {True: "ai", False: "non_ai"}.get(rec.get("is_ai_tool"), "unknown"),
                      "source_category": cat.get("category") or ""})
    return items


def kaggle_items(rng):
    used = {r["source_id"] for r in rows(JEV_EVAL)}
    test = {r["source_id"] for r in rows(MODEL_DIR / "split_manifest.csv") if r["partition"] == "test"}
    pool = {}
    for r in rows(PREPARED / "training_rows.csv"):
        if r["source_id"] in test and r["source_id"] not in used and r["description_only_text"].strip():
            pool.setdefault(r["category"], []).append(r)
    items = []
    for cat in sorted(pool):
        picked = sorted(pool[cat], key=lambda r: r["source_id"])
        rng.shuffle(picked)
        for r in picked[:PER_CATEGORY]:
            items.append({"source": "kaggle_test_split", "source_id": r["source_id"], "tool_name": r["tool_name"],
                          "website": r.get("authority") or "", "text": r["description_only_text"],
                          "text_source": "catalog description", "source_is_ai": "ai", "source_category": cat})
    return items


def non_ai_items(reference, rng, exclude):
    pool = [r for r in reference if r["is_ai_tool"].strip().lower() in ("false", "0")
            and r.get("description", "").strip() and norm(r["tool_name"]) not in exclude]
    pool.sort(key=lambda r: r["tool_id"])
    rng.shuffle(pool)
    return [{"source": "non_ai_reference", "source_id": r["tool_id"], "tool_name": r["tool_name"],
             "website": r.get("domain") or "", "text": build_model_text(r["description"], mode="description_only"),
             "text_source": "reference description", "source_is_ai": "non_ai", "source_category": ""}
            for r in pool[:NON_AI]]


def jev_predict(items, enabled):
    taxonomy = json.loads((MODEL_DIR / "taxonomy.json").read_text(encoding="utf-8"))["categories"]
    criteria = {slug(c): t for c, t in json.loads(CRITERIA.read_text(encoding="utf-8")).items()}
    by_slug = {slug(c): c for c in taxonomy}
    cache = json.loads(JEV_CACHE.read_text(encoding="utf-8")) if JEV_CACHE.exists() else {"version": VERSION, "entries": {}}

    def key(text):
        spec = {"v": VERSION, "model": "jev-latest", "instructions": INSTRUCTIONS, "criteria": criteria, "evidence": text}
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    todo = [it for it in items if it["text"].strip() and key(it["text"]) not in cache["entries"]]
    print(f"Jev: {len(items) - len(todo)} cached, {len(todo)} new calls {'needed' if enabled else 'skipped (--no-jev)'}")
    if todo and enabled:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise SystemExit("Set TYPESAFE_API_KEY in .env, or use --no-jev.")
        for i, it in enumerate(todo, 1):
            res = call_jev("jev-latest", it["text"], criteria, 60, api_key)
            cache["entries"][key(it["text"])] = {"source": "evaluation.build_trusted_test_set", "result": res,
                                                 "retrieved_at": datetime.now(timezone.utc).isoformat()}
            if i % 20 == 0 or i == len(todo):
                write_json(JEV_CACHE, cache)
                print(f"  {i}/{len(todo)}")
    out = {}
    for it in items:
        e = cache["entries"].get(key(it["text"])) if it["text"].strip() else None
        if e:
            r = e["result"]
            out[it["item_id"]] = (by_slug[r["choice"]], round(r["probabilities"][r["choice"]], 4))
    return out


def write_sheet(items, taxonomy, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "Label here"
    head = ["item_id", "tool_name", "website", "description", "is_ai", "category", "confidence", "notes"]
    ws.append(head)
    for it in items:
        ws.append([it["item_id"], it["tool_name"], it["website"], it["text"], "", "", "", ""])
    widths = {"A": 9, "B": 24, "C": 26, "D": 80, "E": 10, "F": 30, "G": 12, "H": 30}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    fill = PatternFill("solid", fgColor="DDEBF7")
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for r in range(2, len(items) + 2):
        ws[f"D{r}"].alignment = Alignment(wrap_text=True, vertical="top")
        for col in "EFG":
            ws[f"{col}{r}"].fill = fill
    ws.freeze_panes = "B2"

    lists = wb.create_sheet("lists")
    options = taxonomy + ["None of these fit", "Not applicable (non-AI)"]
    for i, v in enumerate(options, 1):
        lists[f"A{i}"] = v
    for i, v in enumerate(["ai", "non_ai", "unsure"], 1):
        lists[f"B{i}"] = v
    for i, v in enumerate(["high", "medium", "low"], 1):
        lists[f"C{i}"] = v
    lists.sheet_state = "hidden"
    last = len(items) + 1
    for col, ref in (("E", "$B$1:$B$3"), ("F", f"$A$1:$A${len(options)}"), ("G", "$C$1:$C$3")):
        dv = DataValidation(type="list", formula1=f"=lists!{ref}", allow_blank=True)
        dv.add(f"{col}2:{col}{last}")
        ws.add_data_validation(dv)

    guide = wb.create_sheet("Guide")
    criteria = json.loads(CRITERIA.read_text(encoding="utf-8"))
    lines = [
        "HOW TO LABEL (about 1-2 hours for 150 tools)",
        "Use only the description and your own knowledge of the product. Do not look up model answers.",
        "Label independently: do not discuss items with the other labeller until both sheets are done.",
        "",
        "is_ai: 'ai' if AI/ML is central to the product's PRIMARY purpose (coding assistants, generation,",
        "AI search, model platforms). 'non_ai' if it is a general product where AI is optional or incidental",
        "(e.g. a CRM or office suite with an AI add-on). 'unsure' if the description is not enough.",
        "",
        "category: for AI tools, the ONE business purpose that fits best. For non-AI tools choose",
        "'Not applicable (non-AI)'. If no category fits, choose 'None of these fit' and say why in notes.",
        "Pick by what the product is FOR, not by how it works: almost every tool here uses AI, so use",
        "'Ai And Machine Learning' only for general AI assistants, model platforms and AI tooling.",
        "",
        "confidence: how sure you are about the category (high / medium / low).",
        "",
        "CATEGORIES (examples are typical Kaggle products in each):",
    ] + [f"- {text}" for text in criteria.values()]
    for i, line in enumerate(lines, 1):
        guide[f"A{i}"] = line
    guide.column_dimensions["A"].width = 120
    wb.move_sheet("Guide", offset=-1)
    wb.active = 0
    wb.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-jev", action="store_true")
    a = p.parse_args()
    if (OUT / "items.csv").exists():
        raise SystemExit(f"{OUT.relative_to(ROOT)} already built; delete it to rebuild.")
    rng = random.Random(SEED)
    catalog = {r["tool_id"]: r.get("description", "") for r in rows(CATALOG)}
    reference = rows(REFERENCE)
    usage = usage_items(catalog, reference)
    items = usage + kaggle_items(rng) + non_ai_items(reference, rng, {norm(i["tool_name"]) for i in usage})
    rng.shuffle(items)                                   # mix sources so labellers cannot tell them apart
    for n, it in enumerate(items, 1):
        it["item_id"] = f"T{n:03d}"
    missing = [it["tool_name"] for it in items if not it["text"].strip()]
    if missing:
        print("No description (labellers judge from name/website):", missing)

    import joblib
    taxonomy = json.loads((MODEL_DIR / "taxonomy.json").read_text(encoding="utf-8"))["categories"]
    est = joblib.load(MODEL_DIR / "tfidf.joblib")["estimator"]
    local = dict(zip([it["item_id"] for it in items],
                     est.predict([it["text"] or it["tool_name"] for it in items])))
    jev = jev_predict(items, not a.no_jev)

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["item_id", "source", "source_id", "tool_name", "website", "text_source",
              "source_is_ai", "source_category", "text"]
    with open(OUT / "items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(items, key=lambda i: i["item_id"]))
    with open(OUT / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "local_tfidf_category", "jev_category", "jev_probability"])
        for it in sorted(items, key=lambda i: i["item_id"]):
            jc, jp = jev.get(it["item_id"], ("", ""))
            w.writerow([it["item_id"], local[it["item_id"]], jc, jp])
    write_sheet(sorted(items, key=lambda i: i["item_id"]), taxonomy, OUT / "labeling_sheet.xlsx")
    counts = {}
    for it in items:
        counts[it["source"]] = counts.get(it["source"], 0) + 1
    print(f"Built {len(items)} items {counts} in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
