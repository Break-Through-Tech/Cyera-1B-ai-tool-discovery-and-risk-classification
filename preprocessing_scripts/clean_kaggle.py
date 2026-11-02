"""
clean_kaggle.py

Stage-1 cleaning of the raw Kaggle AI-tools file.
Read-only on the raw file; writes a cleaned working dataset plus logs.

Usage (from the project root):
    python clean_kaggle.py                 # flag dead sites, keep every row
    python clean_kaggle.py --drop-dead     # also remove rows with dead websites

What it does
    1. Keeps only the columns we need, renames them toward the VISION.md schema
       (Year Founded, Country, industry, technologies, Logo are dropped).
    2. Repairs mojibake (e.g. "â€™" -> "’") and normalizes whitespace.
    3. Cleans websites: strips the ?ref=aitoolbuzz.com tracking suffix and
       extracts a lowercase `domain` (no scheme, no www.).
    4. Groups Website Status into active / blocked / dead / uncertain / unknown.
    5. Adds quality and duplicate flags (nothing is dropped for these).
    6. Builds `text_for_model` (name + description + primary task + keywords),
       the combined text to feed embeddings / an LLM classifier.
    7. Assigns a stable tool_id from the raw row order (tool_00001 = first row).

Outputs
    data/processed/kaggle_clean.csv           (UTF-8 with BOM so Excel reads it correctly)
    data/reports/clean_kaggle_log.csv        (every text repair and every dropped row)
    data/reports/clean_kaggle_summary.txt    (counts and the status-grouping table)

Optional: `pip install ftfy` gives better mojibake repair. A built-in fallback
is used if it is not installed.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

try:
    from ftfy import fix_encoding as _ftfy_fix_encoding  # fixes mojibake only
except ImportError:
    _ftfy_fix_encoding = None

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "kaggle.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "kaggle_clean.csv"
REPORT_DIR = REPO_ROOT / "data" / "reports"

# Columns we keep (raw name -> working name). Everything else is dropped.
RENAME = {
    "Name": "tool_name",
    "Website": "website_url_raw",
    "Short Description": "description",
    "Category": "category_raw",
    "Primary Task": "primary_task",
    "Keywords": "keywords",
    "Website Status": "website_status",
    "Link": "source_url",
}
INTENTIONALLY_DROPPED = ["Logo", "Year Founded", "Country", "industry", "technologies"]

MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-"}
MOJIBAKE_RE = re.compile(r"â€|Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|ï¿½|\ufffd")
REF_RE = re.compile(r"[?&]ref=aitoolbuzz\.com/?", re.IGNORECASE)
DOMAIN_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")

# Hosts shared by many unrelated products: the domain alone doesn't identify the tool.
PLATFORM_SUFFIXES = (
    ".vercel.app", ".web.app", ".firebaseapp.com", ".netlify.app", ".github.io",
    ".gumroad.com", ".softr.app", ".herokuapp.com", ".pages.dev", ".streamlit.app",
    ".hf.space", ".notion.site", ".carrd.co", ".wixsite.com", ".framer.app",
    ".framer.website", ".webflow.io", ".onrender.com", ".replit.app",
    ".bubbleapps.io", ".glitch.me", ".blogspot.com", ".wordpress.com",
)
PLATFORM_EXACT = {
    "github.com", "huggingface.co", "play.google.com", "apps.apple.com",
    "chromewebstore.google.com", "chrome.google.com",
}

BLOCKED_CODES = {401, 403, 406, 429, 436, 999}  # bot-blocking; site probably alive
DEAD_CODES = {404, 410}

LOG = []
SUMMARY = []


def say(text=""):
    print(text)
    SUMMARY.append(str(text))


# --------------------------------------------------------------------------- #
# Loading and basic normalization
# --------------------------------------------------------------------------- #
def load_raw(path):
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            if enc != "utf-8-sig":
                print(f"WARNING: {path} is not valid UTF-8; read as {enc}.")
            return df
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Could not decode {path}")


def to_na(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip()
    return pd.NA if s.lower() in MISSING_TOKENS else s


def norm_ws(v):
    if pd.isna(v):
        return v
    s = unicodedata.normalize("NFC", str(v))
    return re.sub(r"\s+", " ", s).strip()


def fix_mojibake(s):
    if not MOJIBAKE_RE.search(s):
        return s
    if _ftfy_fix_encoding is not None:
        fixed = _ftfy_fix_encoding(s)
        if fixed != s:
            return fixed
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def repair_text(df, cols):
    for col in cols:
        has_bad = df[col].fillna("").str.contains(MOJIBAKE_RE)
        for idx in df.index[has_bad]:
            before = df.at[idx, col]
            after = fix_mojibake(before)
            still_bad = bool(MOJIBAKE_RE.search(after))
            df.at[idx, col] = after
            LOG.append({
                "tool_id": df.at[idx, "tool_id"],
                "tool_name": df.at[idx, "tool_name"],
                "action": "mojibake_unrecoverable" if still_bad else "mojibake_fixed",
                "field": col,
                "before": before,
                "after": after,
            })
        df[col] = df[col].map(norm_ws)
    return df


# --------------------------------------------------------------------------- #
# Websites
# --------------------------------------------------------------------------- #
def clean_url(u):
    if pd.isna(u):
        return pd.NA
    u = REF_RE.sub("", str(u).strip())
    if "?" not in u and "&" in u:      # "?ref=...&a=1" leaves a dangling "&a=1"
        u = u.replace("&", "?", 1)
    u = u.rstrip("#/")
    return u or pd.NA


def parse_url(u):
    """Return (domain, path) from a cleaned URL."""
    if pd.isna(u):
        return pd.NA, pd.NA
    full = u if "://" in u else "https://" + u
    try:
        p = urlparse(full)
        host = (p.hostname or "").lower()
    except ValueError:
        return pd.NA, pd.NA
    host = re.sub(r"^www\.", "", host)
    return (host or pd.NA), (p.path.rstrip("/") or "")


def is_platform_host(domain):
    if pd.isna(domain):
        return False
    return domain in PLATFORM_EXACT or domain.endswith(PLATFORM_SUFFIXES)


def status_group(s):
    if pd.isna(s):
        return "unknown"
    t = str(s).lower()
    if t == "active":
        return "active"
    if "not reachable" in t or "expired" in t:
        return "dead"
    m = re.search(r"\b(\d{3})\b", t)
    if m:
        code = int(m.group(1))
        if code in BLOCKED_CODES:
            return "blocked"
        if code in DEAD_CODES:
            return "dead"
    return "uncertain"


# --------------------------------------------------------------------------- #
# Flags and combined text
# --------------------------------------------------------------------------- #
def description_quality(d):
    if pd.isna(d):
        return "missing"
    n = len(d)
    if n < 40:
        return "very_short"
    if n < 80:
        return "short"
    return "ok"


def build_text(row, max_keywords=20):
    parts = [str(row["tool_name"])]
    if pd.notna(row["description"]):
        parts.append(f"Description: {row['description']}")
    if pd.notna(row["primary_task"]):
        parts.append(f"Primary task: {row['primary_task']}")
    if pd.notna(row["keywords"]):
        kws = [k.strip() for k in str(row["keywords"]).split(",") if k.strip()]
        parts.append("Keywords: " + ", ".join(kws[:max_keywords]))
    return " | ".join(parts)


def count_dupes(series):
    counts = series.map(series.value_counts())
    return counts.fillna(0).astype(int)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Clean the raw Kaggle AI-tools CSV")
    ap.add_argument("--path", type=Path, default=RAW_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--drop-dead", action="store_true",
                    help="remove rows whose website is dead (Not Reachable / Expired / 404 / 410)")
    args = ap.parse_args()

    def resolve(path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    args.path = resolve(args.path)
    args.out = resolve(args.out)
    args.report_dir = resolve(args.report_dir)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not args.path.exists():
        sys.exit(f"File not found: {args.path}")

    raw = load_raw(args.path)
    missing = [c for c in RENAME if c not in raw.columns]
    if missing:
        sys.exit(f"Expected columns not found in the raw file: {missing}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    n_in = len(raw)
    df = raw.apply(lambda col: col.map(to_na))
    df["raw_row"] = range(1, n_in + 1)
    df["tool_id"] = df["raw_row"].map(lambda i: f"tool_{i:05d}")
    dropped_cols = [c for c in raw.columns if c not in RENAME]
    df = df[list(RENAME) + ["raw_row", "tool_id"]].rename(columns=RENAME)

    # 1. Text repair
    df = repair_text(df, ["tool_name", "description", "primary_task", "keywords", "category_raw"])

    # 2. Websites
    df["website_url_clean"] = df["website_url_raw"].map(clean_url)
    parsed = df["website_url_clean"].map(parse_url)
    df["domain"] = parsed.map(lambda t: t[0])
    df["url_path"] = parsed.map(lambda t: t[1])
    df["is_platform_host"] = df["domain"].map(is_platform_host)
    df["domain_valid"] = df["domain"].map(lambda d: bool(pd.notna(d) and DOMAIN_RE.match(d)))
    df["site_status_group"] = df["website_status"].map(status_group)

    # 3. Flags
    df["description_quality"] = df["description"].map(description_quality)
    norm_name = df["tool_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    df["name_dup_count"] = count_dupes(norm_name)
    df["domain_dup_count"] = count_dupes(df["domain"])

    # 4. Combined text
    df["text_for_model"] = df.apply(build_text, axis=1)

    # 5. Optional row drops
    if args.drop_dead:
        dead = df["site_status_group"] == "dead"
        for _, r in df[dead].iterrows():
            LOG.append({
                "tool_id": r["tool_id"], "tool_name": r["tool_name"],
                "action": "row_dropped_dead_site", "field": "website_status",
                "before": r["website_status"], "after": "",
            })
        df = df[~dead].copy()

    out_cols = [
        "tool_id", "tool_name", "domain", "description", "category_raw",
        "primary_task", "keywords", "website_status", "site_status_group",
        "website_url_clean", "url_path", "source_url", "is_platform_host",
        "domain_valid", "description_quality", "name_dup_count",
        "domain_dup_count", "text_for_model", "raw_row",
    ]
    df = df[out_cols]
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    pd.DataFrame(LOG, columns=["tool_id", "tool_name", "action", "field", "before", "after"]).to_csv(
        args.report_dir / "clean_kaggle_log.csv", index=False, encoding="utf-8-sig"
    )

    # ---------------------------------------------------------------- summary
    say("=" * 70)
    say("CLEAN_KAGGLE SUMMARY")
    say("=" * 70)
    say(f"Rows in:  {n_in:,}")
    say(f"Rows out: {len(df):,}   (dropped {n_in - len(df):,}{' - dead sites' if args.drop_dead else ''})")
    say(f"Columns dropped on purpose: {[c for c in dropped_cols if c in INTENTIONALLY_DROPPED]}")
    other = [c for c in dropped_cols if c not in INTENTIONALLY_DROPPED]
    if other:
        say(f"Other unexpected columns also dropped: {other}")
    say()

    log_df = pd.DataFrame(LOG)
    if len(log_df):
        say("Text repairs / drops:")
        for action, n in log_df["action"].value_counts().items():
            say(f"  {action}: {n:,}")
    else:
        say("Text repairs / drops: none")
    say()

    say("Website status -> group (check this mapping against your data):")
    table = (df.groupby(["site_status_group", df["website_status"].fillna("<missing>")])
               .size().rename("rows").reset_index())
    say(table.to_string(index=False))
    say()

    say("Description quality:")
    for k, v in df["description_quality"].value_counts().items():
        say(f"  {k:<11} {v:>6,}")
    say()
    say(f"Rows with a shared/platform host (domain isn't a unique identifier): {int(df['is_platform_host'].sum()):,}")
    say(f"Rows with an invalid or missing domain: {int((~df['domain_valid']).sum()):,}")
    say(f"Rows sharing a domain with another row: {int((df['domain_dup_count'] > 1).sum()):,}")
    say(f"Rows sharing a normalized name with another row: {int((df['name_dup_count'] > 1).sum()):,}")
    say(f"tool_id unique: {df['tool_id'].is_unique}")
    say()
    say(f"Wrote {args.out}")
    say(f"Wrote {args.report_dir / 'clean_kaggle_log.csv'}")

    (args.report_dir / "clean_kaggle_summary.txt").write_text("\n".join(SUMMARY), encoding="utf-8")


if __name__ == "__main__":
    main()