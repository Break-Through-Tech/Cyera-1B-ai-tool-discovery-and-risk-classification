"""
Finalize Dataset B by filling blank vendor + synthetic governance columns.

Reads:  data/dataset_b_ai_tool_metadata_incomplete.csv
Writes: data/dataset_b_ai_tool_metadata.csv

Vendors use a two-tier local heuristic (known-domain map, then domain/name parse).
Governance attributes are correlated synthetic fields; risk_label is rule-derived.
No external API calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = REPO_ROOT / "data" / "dataset_b_ai_tool_metadata_incomplete.csv"
OUTPUT_FILE = REPO_ROOT / "data" / "dataset_b_ai_tool_metadata.csv"

SCHEMA_ORDER = [
    "tool_id",
    "tool_name",
    "domain",
    "description",
    "is_ai_tool",
    "category",
    "vendor",
    "approved_status",
    "enterprise_plan_available",
    "sso_available",
    "data_retention",
    "training_on_customer_data",
    "admin_controls_available",
    "risk_label",
]

# Well-known domains → (vendor, typically approved for enterprise use)
KNOWN_VENDORS: dict[str, tuple[str, bool]] = {
    "chatgpt.com": ("OpenAI", True),
    "openai.com": ("OpenAI", True),
    "chat.openai.com": ("OpenAI", True),
    "claude.ai": ("Anthropic", True),
    "anthropic.com": ("Anthropic", True),
    "gemini.google.com": ("Google", True),
    "bard.google.com": ("Google", True),
    "ai.google": ("Google", True),
    "copilot.microsoft.com": ("Microsoft", True),
    "github.com": ("Microsoft", True),
    "cursor.com": ("Anysphere", True),
    "midjourney.com": ("Midjourney", True),
    "grammarly.com": ("Grammarly", True),
    "jasper.ai": ("Jasper", True),
    "notion.so": ("Notion", True),
    "notion.com": ("Notion", True),
    "perplexity.ai": ("Perplexity", True),
    "huggingface.co": ("Hugging Face", True),
    "replicate.com": ("Replicate", False),
    "stability.ai": ("Stability AI", False),
    "cohere.com": ("Cohere", True),
    "mistral.ai": ("Mistral AI", True),
    "together.ai": ("Together AI", False),
    "anyscale.com": ("Anyscale", False),
    "adobe.com": ("Adobe", True),
    "canva.com": ("Canva", True),
    "figma.com": ("Figma", True),
    "otter.ai": ("Otter.ai", True),
    "fireflies.ai": ("Fireflies.ai", False),
    "writesonic.com": ("Writesonic", False),
    "copy.ai": ("Copy.ai", False),
    "character.ai": ("Character.AI", False),
    "poe.com": ("Quora", False),
    "groq.com": ("Groq", False),
    "deepmind.google": ("Google", True),
}

# Longest-first so multi-part suffixes match before short ones.
_DOMAIN_SUFFIXES = (
    ".co.uk",
    ".com.au",
    ".com.br",
    ".co.jp",
    ".co.in",
    ".com",
    ".org",
    ".net",
    ".edu",
    ".gov",
    ".io",
    ".ai",
    ".co",
    ".so",
    ".dev",
    ".app",
    ".gg",
    ".xyz",
    ".tech",
    ".me",
    ".info",
    ".biz",
    ".tv",
    ".cc",
    ".us",
    ".uk",
    ".ca",
    ".de",
    ".fr",
    ".nl",
    ".eu",
    ".cloud",
    ".online",
    ".site",
    ".studio",
    ".tools",
    ".space",
    ".pro",
    ".sh",
    ".to",
    ".ly",
    ".fm",
    ".ms",
    ".vercel.app",
    ".herokuapp.com",
    ".github.io",
    ".pages.dev",
)


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _ensure_str_col(df: pd.DataFrame, col: str) -> None:
    """Coerce a column to string dtype; blank/NaN CSV fields often load as float64."""
    if col not in df.columns:
        df[col] = ""
    else:
        df[col] = df[col].fillna("").astype(str)
        df.loc[df[col].str.lower().isin(["nan", "none"]), col] = ""


def vendor_from_domain_or_name(domain: str, tool_name: str) -> str:
    """
    Tier-2 heuristic: strip TLDs from the domain and capitalize the root label.

    Examples:
      example-ai.io      -> Example-ai
      sub.writesonic.com -> Writesonic
      (empty domain)     -> tool_name (or Unknown)
    """
    host = str(domain or "").strip().lower()
    host = host.split("/")[0].split(":")[0]
    if host.startswith("www."):
        host = host[4:]

    fallback = str(tool_name or "").strip() or "Unknown"
    if not host or host in {"unknown", "nan", "none"}:
        return fallback

    root = host
    for suffix in _DOMAIN_SUFFIXES:
        if host.endswith(suffix) and len(host) > len(suffix):
            root = host[: -len(suffix)]
            break

    # Drop remaining subdomains: a.b.brand -> brand
    if "." in root:
        root = root.rsplit(".", 1)[-1]

    root = root.strip(".-_")
    if not root:
        return fallback

    return root[0].upper() + root[1:]


def apply_known_vendors(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 1: fill vendor for curated well-known domains."""
    df = df.copy()
    _ensure_str_col(df, "vendor")

    domains = df["domain"].fillna("").astype(str).str.lower().str.strip()
    blank = _is_blank(df["vendor"])

    for domain, (vendor, _) in KNOWN_VENDORS.items():
        mask = blank & (domains == domain)
        df.loc[mask, "vendor"] = vendor

    return df


def apply_heuristic_vendors(df: pd.DataFrame) -> pd.DataFrame:
    """Tier 2: fill any remaining blank vendors from domain/tool_name."""
    df = df.copy()
    _ensure_str_col(df, "vendor")

    blank = _is_blank(df["vendor"])
    if not blank.any():
        return df

    domains = df.loc[blank, "domain"].fillna("").astype(str)
    names = df.loc[blank, "tool_name"].fillna("").astype(str)
    df.loc[blank, "vendor"] = [
        vendor_from_domain_or_name(d, n) for d, n in zip(domains, names)
    ]
    return df


def _is_major_tool(row: pd.Series) -> bool:
    domain = str(row.get("domain", "")).lower().strip()
    if domain in KNOWN_VENDORS:
        return KNOWN_VENDORS[domain][1]
    name = str(row.get("tool_name", "")).lower()
    major_tokens = (
        "chatgpt",
        "openai",
        "claude",
        "anthropic",
        "gemini",
        "copilot",
        "cursor",
        "midjourney",
        "grammarly",
        "jasper",
        "perplexity",
        "huggingface",
        "mistral",
        "cohere",
    )
    return any(tok in name for tok in major_tokens)


def calculate_risk(row: pd.Series) -> str:
    """Explainable multi-factor risk label (ground truth for Dataset B)."""
    score = 0

    if row["approved_status"] != "approved":
        score += 2
    if bool(row["training_on_customer_data"]):
        score += 2
    if row["data_retention"] == "indefinite":
        score += 1
    if not bool(row["sso_available"]):
        score += 1
    if not bool(row["enterprise_plan_available"]):
        score += 1
    if not bool(row["admin_controls_available"]):
        score += 1

    # Approved + strong controls can still be medium if retention/training are weak.
    if score <= 1:
        return "low"
    if score <= 3:
        return "medium"
    return "high"


def generate_synthetic_governance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate governance fields with correlated probabilities.

    Major / known tools skew approved with enterprise controls.
    Enterprise-plan tools usually also get SSO + admin controls and less
    aggressive training-on-customer-data defaults.
    """
    print("Generating synthetic governance and security metrics...")
    rng = np.random.default_rng(42)
    df = df.copy()
    n = len(df)

    major = df.apply(_is_major_tool, axis=1).to_numpy()

    approved = np.empty(n, dtype=object)
    enterprise = np.empty(n, dtype=bool)
    sso = np.empty(n, dtype=bool)
    admin = np.empty(n, dtype=bool)
    training = np.empty(n, dtype=bool)
    retention = np.empty(n, dtype=object)

    for i in range(n):
        is_major = bool(major[i])

        # Approval: major tools usually approved; long-tail mostly shadow AI.
        approve_p = 0.85 if is_major else 0.18
        is_approved = rng.random() < approve_p
        approved[i] = "approved" if is_approved else "unapproved"

        # Enterprise plan more common when approved / major.
        if is_major or is_approved:
            ent_p = 0.80
        else:
            ent_p = 0.25
        has_enterprise = rng.random() < ent_p
        enterprise[i] = has_enterprise

        # SSO and admin controls correlated with enterprise plan.
        if has_enterprise:
            sso[i] = rng.random() < 0.90
            admin[i] = rng.random() < 0.85
            train_p = 0.25
            retention_p = [0.35, 0.30, 0.15, 0.20]  # 30d, 90d, indefinite, custom
        else:
            sso[i] = rng.random() < 0.20
            admin[i] = rng.random() < 0.15
            train_p = 0.70
            retention_p = [0.20, 0.25, 0.40, 0.15]

        training[i] = rng.random() < train_p
        retention[i] = rng.choice(
            ["30_days", "90_days", "indefinite", "custom_policy"],
            p=retention_p,
        )

    df["approved_status"] = approved
    df["enterprise_plan_available"] = enterprise
    df["sso_available"] = sso
    df["admin_controls_available"] = admin
    df["training_on_customer_data"] = training
    df["data_retention"] = retention
    df["risk_label"] = df.apply(calculate_risk, axis=1)
    return df


def validate_input(df: pd.DataFrame) -> None:
    required = {"tool_name", "domain", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")


def main() -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"Error: Input file '{INPUT_FILE}' not found. "
            "Run convert_kaggle.py first."
        )

    df = pd.read_csv(INPUT_FILE)
    validate_input(df)
    print(f"Loaded {len(df)} records from {INPUT_FILE.relative_to(REPO_ROOT)}.")

    for col in SCHEMA_ORDER:
        if col not in df.columns:
            df[col] = "" if col != "is_ai_tool" else True

    df = apply_known_vendors(df)
    known_filled = (~_is_blank(df["vendor"])).sum()
    print(f"Tier 1 known vendors filled: {known_filled}")

    df = apply_heuristic_vendors(df)
    print(f"Vendors complete: {(~_is_blank(df['vendor'])).sum()} / {len(df)}")

    df = generate_synthetic_governance(df)

    df = df[SCHEMA_ORDER]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Success! Enriched dataset saved to '{OUTPUT_FILE.relative_to(REPO_ROOT)}'.")
    print(df["approved_status"].value_counts().to_string())
    print(df["risk_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
