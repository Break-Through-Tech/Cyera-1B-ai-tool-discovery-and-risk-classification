"""Score models against the human-labelled trusted test set.

    python evaluation/score_trusted_set.py --labels min.xlsx
    python evaluation/score_trusted_set.py --labels min.xlsx teammate.xlsx
    python evaluation/score_trusted_set.py --labels min.xlsx teammate.xlsx --adjudicated settled.xlsx

With two labellers it reports agreement (Cohen's kappa) and uses only items both
agree on as ground truth; disagreements are written to disagreements.csv. Fill those
rows in a copy of the sheet and pass it as --adjudicated to settle them.

Writes evaluation/trusted_test_set/report.md and results.json.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "evaluation/trusted_test_set"
NONE_FIT = "None of these fit"
NOT_APPLICABLE = "Not applicable (non-AI)"


def read_labels(path: Path) -> dict:
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        ws = load_workbook(path, read_only=True, data_only=True)["Label here"]
        it = ws.iter_rows(values_only=True)
        head = [str(h) for h in next(it)]
        data = [dict(zip(head, r)) for r in it]
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
    out = {}
    for r in data:
        iid = (r.get("item_id") or "").strip()
        is_ai = (r.get("is_ai") or "").strip().lower()
        cat = (r.get("category") or "").strip()
        if iid and (is_ai or cat):
            out[iid] = {"is_ai": is_ai, "category": cat,
                        "confidence": (r.get("confidence") or "").strip().lower()}
    return out


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def mcnemar(right_a, right_b):
    b = sum(x and not y for x, y in zip(right_a, right_b))
    c = sum(y and not x for x, y in zip(right_a, right_b))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"a_right_b_wrong": b, "b_right_a_wrong": c, "p_value": p}


def scores(gold, pred, labels):
    from sklearn.metrics import accuracy_score, f1_score
    return {"n": len(gold), "accuracy": round(accuracy_score(gold, pred), 4),
            "macro_f1": round(f1_score(gold, pred, labels=labels, average="macro", zero_division=0), 4)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, nargs="+", required=True, help="Filled labelling sheets (.xlsx or .csv)")
    p.add_argument("--adjudicated", type=Path, help="Sheet settling disagreements between labellers")
    a = p.parse_args()

    with open(SET / "items.csv", newline="", encoding="utf-8") as f:
        items = {r["item_id"]: r for r in csv.DictReader(f)}
    with open(SET / "predictions.csv", newline="", encoding="utf-8") as f:
        preds = {r["item_id"]: r for r in csv.DictReader(f)}
    taxonomy = json.loads((ROOT / "artifacts/jev_category_eval/criteria.json").read_text(encoding="utf-8"))
    taxonomy = list(taxonomy)
    sheets = [read_labels(x) for x in a.labels]
    for path, s in zip(a.labels, sheets):
        if not s:
            raise SystemExit(f"{path} has no labels filled in yet.")
    results = {"labellers": [x.name for x in a.labels], "items_total": len(items),
               "labelled_per_sheet": [len(s) for s in sheets]}

    # ---- agreement and ground truth
    gold, disputes = {}, []
    if len(sheets) == 1:
        gold = sheets[0]
    else:
        common = sorted(set.intersection(*(set(s) for s in sheets)))
        ai_a = [sheets[0][i]["is_ai"] for i in common]
        ai_b = [sheets[1][i]["is_ai"] for i in common]
        both_ai = [i for i in common if all(s[i]["is_ai"] == "ai" for s in sheets)]
        cat_a = [sheets[0][i]["category"] for i in both_ai]
        cat_b = [sheets[1][i]["category"] for i in both_ai]
        results["agreement"] = {
            "items_labelled_by_all": len(common),
            "is_ai_percent": round(sum(x == y for x, y in zip(ai_a, ai_b)) / len(common), 4) if common else None,
            "is_ai_kappa": kappa(ai_a, ai_b),
            "category_percent_when_both_ai": round(sum(x == y for x, y in zip(cat_a, cat_b)) / len(both_ai), 4) if both_ai else None,
            "category_kappa_when_both_ai": kappa(cat_a, cat_b)}
        for i in common:
            labs = [s[i] for s in sheets]
            if all(l["is_ai"] == labs[0]["is_ai"] and l["category"] == labs[0]["category"] for l in labs):
                gold[i] = labs[0]
            else:
                disputes.append(i)
        if a.adjudicated:
            settled = read_labels(a.adjudicated)
            for i in list(disputes):
                if i in settled:
                    gold[i] = settled[i]
                    disputes.remove(i)
        with open(SET / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "tool_name"] + [f"{x.stem}_{k}" for x in a.labels for k in ("is_ai", "category")])
            for i in disputes:
                w.writerow([i, items[i]["tool_name"]] + [s[i][k] for s in sheets for k in ("is_ai", "category")])
        results["unresolved_disagreements"] = len(disputes)

    # ---- AI detection: source labels vs. humans
    det = [i for i, g in gold.items() if g["is_ai"] in ("ai", "non_ai")]
    truth = [gold[i]["is_ai"] == "ai" for i in det]
    src = [items[i]["source_is_ai"] == "ai" for i in det]
    tp = sum(t and s for t, s in zip(truth, src))
    fp = sum(s and not t for t, s in zip(truth, src))
    fn = sum(t and not s for t, s in zip(truth, src))
    results["detection_source_labels"] = {
        "n": len(det), "accuracy": round(sum(t == s for t, s in zip(truth, src)) / len(det), 4) if det else None,
        "ai_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "ai_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "source_said_ai_humans_said_non_ai": [items[i]["tool_name"] for i, t, s in zip(det, truth, src) if s and not t],
        "unsure": sum(g["is_ai"] == "unsure" for g in gold.values())}

    # ---- categories
    ai_items = [i for i in det if gold[i]["is_ai"] == "ai"]
    results["category_none_fit"] = sum(gold[i]["category"] == NONE_FIT for i in ai_items)
    cat_ids = [i for i in ai_items if gold[i]["category"] in taxonomy]
    g = [gold[i]["category"] for i in cat_ids]
    local = [preds[i]["local_tfidf_category"] for i in cat_ids]
    jev = [preds[i]["jev_category"] for i in cat_ids]
    results["category"] = {"local_tfidf": scores(g, local, taxonomy) if g else None,
                           "jev": scores(g, jev, taxonomy) if g else None}
    kag = [i for i in cat_ids if items[i]["source"] == "kaggle_test_split"]
    if kag:
        results["category"]["kaggle_label_on_kaggle_items"] = scores(
            [gold[i]["category"] for i in kag], [items[i]["source_category"] for i in kag], taxonomy)
    results["category"]["jev_vs_local_mcnemar"] = mcnemar(
        [x == y for x, y in zip(g, jev)], [x == y for x, y in zip(g, local)])
    buckets = {}
    for lo, hi in ((0, .5), (.5, .7), (.7, 1.01)):
        sel = [(x, y) for i, x, y in zip(cat_ids, g, jev) if lo <= float(preds[i]["jev_probability"] or 0) < hi]
        buckets[f"{lo}-{min(hi, 1)}"] = {"n": len(sel), "jev_accuracy": round(sum(x == y for x, y in sel) / len(sel), 4) if sel else None}
    results["category"]["jev_by_probability"] = buckets

    (SET / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    c = results["category"]
    lines = [
        "# Trusted test set results", "",
        f"Labellers: {', '.join(results['labellers'])}. Items: {results['items_total']}.", ""]
    def num(v, f):
        return "n/a" if v is None else format(v, f)

    if "agreement" in results:
        ag = results["agreement"]
        lines += ["## Labeller agreement", "",
                  f"- AI vs non-AI: {num(ag['is_ai_percent'], '.0%')} agreement, "
                  f"Cohen's kappa {num(ag['is_ai_kappa'], '.2f')}",
                  f"- Category (both said AI): {num(ag['category_percent_when_both_ai'], '.0%')} agreement, "
                  f"kappa {num(ag['category_kappa_when_both_ai'], '.2f')}",
                  f"- Unresolved disagreements: {results['unresolved_disagreements']} (see disagreements.csv)", ""]
    d = results["detection_source_labels"]
    lines += ["## AI detection (source labels vs. people)", "",
              f"- {d['n']} items: accuracy {num(d['accuracy'], '.0%')}, AI precision {d['ai_precision']}, AI recall {d['ai_recall']}",
              f"- Labelled AI by the source but non-AI by people: {', '.join(d['source_said_ai_humans_said_non_ai']) or 'none'}",
              "", "## Category accuracy against people", "",
              "| Method | Items | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"]
    names = {"jev": "Jev", "local_tfidf": "Local TF-IDF model", "kaggle_label_on_kaggle_items": "Kaggle's own label (Kaggle items)"}
    for k, label in names.items():
        if c.get(k):
            lines.append(f"| {label} | {c[k]['n']} | {c[k]['accuracy']:.1%} | {c[k]['macro_f1']:.3f} |")
    m = c["jev_vs_local_mcnemar"]
    lines += ["", f"Jev right / local wrong: {m['a_right_b_wrong']}; local right / Jev wrong: "
              f"{m['b_right_a_wrong']}; McNemar p = {m['p_value']:.4f}.", "",
              "Jev accuracy by its own probability: " + "; ".join(
                  f"{k}: {v['jev_accuracy']:.0%} of {v['n']}" for k, v in c["jev_by_probability"].items() if v["n"]),
              "", f"AI tools for which no category fit: {results['category_none_fit']}."]
    (SET / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
