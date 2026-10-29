"""
analyze_kaggle.py

Exploratory data analysis for the raw Kaggle AI-tools file.
Read-only: it never modifies data/raw/kaggle.csv.

Usage (from the project root):
    python analyze_kaggle.py
    python analyze_kaggle.py --path data/raw/kaggle.csv --out data/reports

Outputs:
    - Printed report in the console
    - data/reports/kaggle_eda.txt          (full text report)
    - data/reports/kaggle_missingness.csv  (per-column missingness)
    - data/reports/kaggle_duplicates.csv   (rows sharing a name or domain)
    - data/reports/kaggle_category_counts.csv
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

DEFAULT_PATH = Path("data") / "raw" / "kaggle.csv"
DEFAULT_OUT = Path("data") / "reports"

MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-"}
MOJIBAKE_RE = re.compile(r"â€|Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|ï¿½|\ufffd")
AI_TERMS_RE = re.compile(
    r"\b(ai|a\.i\.|artificial intelligence|machine learning|llm|gpt|"
    r"language model|neural|generative|copilot|chatbot|deep learning)\b",
    re.IGNORECASE,
)

LINES = []


def out(text=""):
    """Print and also keep for the saved report."""
    print(text)
    LINES.append(str(text))


def section(title):
    out()
    out("=" * 78)
    out(title)
    out("=" * 78)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def detect_sep(path, encoding):
    with open(path, "r", encoding=encoding, errors="replace") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def load(path):
    """Try several encodings. Returns (df, encoding, sep)."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            sep = detect_sep(path, enc)
            df = pd.read_csv(
                path,
                sep=sep,
                encoding=enc,
                dtype=str,
                keep_default_na=False,
                on_bad_lines="warn",
            )
            return df, enc, sep
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def normalize_missing(df):
    """Turn blanks / whitespace / 'nan' / 'n/a' etc. into real NA values."""
    def clean(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip()
        return pd.NA if s.lower() in MISSING_TOKENS else s

    return df.apply(lambda col: col.map(clean))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def extract_domain(url):
    if pd.isna(url):
        return pd.NA
    u = str(url).strip()
    if "://" not in u:
        u = "https://" + u
    host = urlparse(u).netloc.lower().split("@")[-1].split(":")[0]
    host = re.sub(r"^www\.", "", host)
    return host or pd.NA


def show_counts(series, n=15, title=None):
    if title:
        out(title)
    vc = series.value_counts(dropna=False).head(n)
    total = len(series)
    for val, cnt in vc.items():
        label = "<missing>" if pd.isna(val) else str(val)
        out(f"  {label[:60]:<60} {cnt:>6}  ({cnt / total * 100:5.1f}%)")


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def overview(df, raw_df, path, enc, sep):
    section("1. FILE OVERVIEW")
    out(f"File:          {path}")
    out(f"Encoding used: {enc}   (a fallback from utf-8 means the file has non-utf-8 bytes)")
    out(f"Delimiter:     {'TAB' if sep == chr(9) else repr(sep)}")
    out(f"Rows:          {len(df):,}")
    out(f"Columns:       {df.shape[1]}")
    out(f"Memory:        {raw_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    out()
    out("Columns:")
    for c in df.columns:
        out(f"  - {c}")


def missingness(df, out_dir):
    section("2. MISSING VALUES (blank, whitespace, 'nan', 'n/a' all count as missing)")
    miss = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "unique_values": df.nunique(dropna=True),
        }
    ).sort_values("pct_missing", ascending=False)
    miss["severity"] = pd.cut(
        miss["pct_missing"],
        bins=[-0.01, 5, 20, 50, 100],
        labels=["ok", "minor (5-20%)", "moderate (20-50%)", "HEAVY (>50%)"],
    )
    miss["severity"] = miss["severity"].astype(str).replace({"ok": "ok (<=5%)"})
    out(miss.to_string())
    miss.to_csv(out_dir / "kaggle_missingness.csv")

    heavy = miss[miss["pct_missing"] > 50].index.tolist()
    out()
    out(f"Heavily missing columns (>50%): {heavy if heavy else 'none'}")

    out()
    out("Missing fields per row (how complete is a typical record?):")
    per_row = df.isna().sum(axis=1)
    show_counts(per_row.rename("missing_fields"), n=15)

    out()
    out("Columns that are fully populated: "
        f"{miss[miss['missing'] == 0].index.tolist() or 'none'}")


def duplicates(df, out_dir):
    section("3. DUPLICATES")
    out(f"Fully duplicated rows: {df.duplicated().sum():,}")

    frames = []
    for col in ("Name", "Link", "Domain"):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        dup_mask = df[col].notna() & df.duplicated(subset=[col], keep=False)
        n_rows = int(dup_mask.sum())
        n_keys = df.loc[dup_mask, col].nunique()
        out(f"Duplicate {col!r}: {n_rows:,} rows share {n_keys:,} repeated values")
        if n_rows:
            d = df.loc[dup_mask, [c for c in ("Name", "Domain", "Category", "Website") if c in df.columns]].copy()
            d.insert(0, "duplicate_on", col)
            frames.append(d)

    if "Name" in df.columns:
        norm = df["Name"].dropna().str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        out(f"Duplicate names after lowercasing/stripping punctuation: "
            f"{norm.duplicated(keep=False).sum():,} rows")

    if frames:
        pd.concat(frames).sort_values(["duplicate_on", "Name"]).to_csv(
            out_dir / "kaggle_duplicates.csv", index=False
        )
        out("Saved duplicate rows to kaggle_duplicates.csv")


def categories(df, out_dir):
    section("4. CATEGORICAL COLUMNS")
    if "Category" in df.columns:
        out(f"Distinct categories: {df['Category'].nunique()}")
        show_counts(df["Category"], n=40, title="Category distribution:")
        df["Category"].value_counts(dropna=False).rename_axis("category").reset_index(
            name="count"
        ).to_csv(out_dir / "kaggle_category_counts.csv", index=False)
        out()

    for col in ("Primary Task", "Country", "industry", "technologies", "Website Status"):
        if col in df.columns:
            out(f"[{col}] distinct={df[col].nunique()}  missing={df[col].isna().sum()}")
            show_counts(df[col], n=10)
            out()


def keywords(df):
    section("5. KEYWORDS")
    if "Keywords" not in df.columns:
        out("No Keywords column.")
        return
    kw = df["Keywords"].dropna().str.split(",").map(lambda xs: [x.strip().lower() for x in xs if x.strip()])
    out(f"Rows with keywords: {len(kw):,} of {len(df):,}")
    if len(kw):
        counts = kw.map(len)
        out(f"Keywords per row: min={counts.min()}  median={int(counts.median())}  max={counts.max()}")
        top = Counter(k for ks in kw for k in ks).most_common(25)
        out("Top 25 keywords:")
        for k, c in top:
            out(f"  {k:<50} {c:>6}")


def text_quality(df):
    section("6. TEXT QUALITY (descriptions, mojibake, AI signal)")
    desc_col = "Short Description"
    if desc_col in df.columns:
        d = df[desc_col].dropna()
        lens = d.str.len()
        out(f"Description length (chars): min={lens.min()}  median={int(lens.median())}  "
            f"mean={lens.mean():.0f}  max={lens.max()}")
        for thresh in (40, 80):
            out(f"  Descriptions shorter than {thresh} chars: {(lens < thresh).sum():,} "
                f"({(lens < thresh).mean() * 100:.1f}%)")
        out(f"  Duplicate descriptions: {d.duplicated(keep=False).sum():,} rows")

        has_ai = d.str.contains(AI_TERMS_RE)
        out(f"  Descriptions with an explicit AI/ML term: {has_ai.sum():,} ({has_ai.mean() * 100:.1f}%)")
        out(f"  Descriptions with NO explicit AI/ML term: {(~has_ai).sum():,} "
            f"(harder for description-based detection)")

        if "Name" in df.columns:
            both = df[[ "Name", desc_col]].dropna()
            names_in = both.apply(
                lambda r: r["Name"].lower() in r[desc_col].lower(), axis=1
            )
            out(f"  Descriptions that mention the tool's own name: {names_in.mean() * 100:.1f}%")

        out()
        out("Shortest 10 descriptions (likely weak signal):")
        short = df.loc[d.index].assign(_len=lens).nsmallest(10, "_len")
        for _, r in short.iterrows():
            out(f"  [{r.get('Name', '?')}] {r[desc_col][:90]}")

    out()
    out("Mojibake / encoding damage (looks like â€™, Ã©, Â, �):")
    found_any = False
    for col in df.select_dtypes(include="object").columns:
        n = df[col].dropna().astype(str).str.contains(MOJIBAKE_RE).sum()
        if n:
            found_any = True
            out(f"  {col}: {n:,} rows")
    if not found_any:
        out("  none found in the raw file")
        out("  -> If your final dataset has mojibake, your pipeline introduced it (re-read with utf-8).")

    out()
    out("Leading/trailing whitespace in text columns:")
    for col in df.select_dtypes(include="object").columns:
        s = df[col].dropna().astype(str)
        n = (s != s.str.strip()).sum()
        if n:
            out(f"  {col}: {n:,} rows")


def websites(df):
    section("7. WEBSITE / DOMAIN CHECKS")
    if "Website" not in df.columns:
        out("No Website column.")
        return
    out(f"Missing Website: {df['Website'].isna().sum():,}")
    out(f"Websites containing '?ref=aitoolbuzz.com' tracking: "
        f"{df['Website'].dropna().str.contains('ref=aitoolbuzz', case=False).sum():,}")
    dom = df["Domain"].dropna()
    out(f"Distinct domains: {dom.nunique():,} of {len(dom):,} rows with a domain")
    show_counts(dom, n=10, title="Most repeated domains:")
    out()
    out("Domains that are not a plain host (spaces, no dot):")
    bad = dom[~dom.str.contains(r"^[a-z0-9.-]+\.[a-z]{2,}$", regex=True)]
    out(f"  {len(bad):,} rows" + (f"  e.g. {bad.head(5).tolist()}" if len(bad) else ""))

    if "Website Status" in df.columns:
        out()
        show_counts(df["Website Status"], n=10, title="Website Status:")
        inactive = df[df["Website Status"].fillna("").str.lower() != "active"]
        out(f"  Rows not marked 'Active': {len(inactive):,} (candidates to drop or flag)")


def year_founded(df):
    section("8. YEAR FOUNDED")
    if "Year Founded" not in df.columns:
        out("No Year Founded column.")
        return
    y = pd.to_numeric(df["Year Founded"], errors="coerce")
    non_numeric = df["Year Founded"].notna() & y.isna()
    out(f"Missing: {df['Year Founded'].isna().sum():,}   Non-numeric: {non_numeric.sum():,}")
    if y.notna().any():
        out(f"Range: {int(y.min())} - {int(y.max())}   median: {int(y.median())}")
        out(f"Implausible (<1950 or >{pd.Timestamp.today().year}): "
            f"{((y < 1950) | (y > pd.Timestamp.today().year)).sum():,}")
        show_counts(y.dropna().astype(int), n=12, title="Most common years:")


def row_completeness_vs_category(df):
    section("9. DATA COMPLETENESS BY CATEGORY (are some categories poorly covered?)")
    if "Category" not in df.columns:
        return
    comp = (
        df.assign(_missing=df.isna().sum(axis=1))
        .groupby("Category")["_missing"]
        .agg(["count", "mean"])
        .rename(columns={"count": "rows", "mean": "avg_missing_fields"})
        .sort_values("avg_missing_fields", ascending=False)
        .round(2)
    )
    out(comp.head(15).to_string())


def summary(df):
    section("10. TAKEAWAYS TO CHECK")
    miss_pct = df.isna().mean() * 100
    heavy = miss_pct[miss_pct > 50].index.tolist()
    out(f"- Columns with >50% missing (probably unusable as features): {heavy or 'none'}")
    if "Category" in df.columns:
        out(f"- {df['Category'].nunique()} raw categories; plan to remap to your own taxonomy.")
    out("- Compare Name/Website/Short Description here against your final dataset with a diff")
    out("  to see which problems are inherited from Kaggle and which your pipeline introduced.")
    out("- Every row in this source is presumably an AI tool, so non-AI negatives must come from elsewhere.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="EDA for the raw Kaggle AI-tools CSV")
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not args.path.exists():
        sys.exit(f"File not found: {args.path.resolve()}\nRun this from the project root.")

    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    raw_df, enc, sep = load(args.path)
    df = normalize_missing(raw_df)
    if "Website" in df.columns:
        df["Domain"] = df["Website"].map(extract_domain)

    overview(df.drop(columns=["Domain"], errors="ignore"), raw_df, args.path, enc, sep)
    missingness(df.drop(columns=["Domain"], errors="ignore"), args.out)
    duplicates(df, args.out)
    categories(df, args.out)
    keywords(df)
    text_quality(df)
    websites(df)
    year_founded(df)
    row_completeness_vs_category(df)
    summary(df)

    report = args.out / "kaggle_eda.txt"
    report.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nSaved report and CSV summaries to {args.out.resolve()}")


if __name__ == "__main__":
    main()