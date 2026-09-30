from urllib.parse import urlparse
import numpy as np
import pandas as pd

INPUT_PATH = "data/cleaned_kaggle.csv"
OUTPUT_PATH = "data/dataset_b_ai_tool_metadata_incomplete.csv"

df = pd.read_csv(INPUT_PATH)

# Helper function to extract clean domain from website URL
# https://chatgpt.com/?ref....  -> chatgpt.com
def extract_domain(url_string):
    if not isinstance(url_string, str) or not url_string.strip():
        return ""
    if not url_string.startswith(("http://", "https://")):
        url_string = "https://" + url_string
    try:
        domain = urlparse(url_string).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.split(":")[0]
    except Exception:
        return ""

dataset_b = pd.DataFrame()

# -----------------------------------------------------------------------------
# Real / Mapped Fields from Kaggle
# -----------------------------------------------------------------------------
dataset_b["tool_id"] = [f"tool_{i+1:05d}" for i in range(len(df))]
dataset_b["tool_name"] = df["Name"].fillna("").astype(str).str.strip()
dataset_b["domain"] = df["Website"].apply(extract_domain)
dataset_b["description"] = (
    df["Short Description"].fillna("").astype(str).str.strip()
)
dataset_b["is_ai_tool"] = True
dataset_b["category"] = df["Category"].fillna("").astype(str).str.strip()

# -----------------------------------------------------------------------------
# Blank Fields (Left empty for separate synthetic generation step later)
# -----------------------------------------------------------------------------
dataset_b["vendor"] = ""
dataset_b["approved_status"] = ""
dataset_b["enterprise_plan_available"] = ""
dataset_b["sso_available"] = ""
dataset_b["data_retention"] = ""
dataset_b["training_on_customer_data"] = ""
dataset_b["admin_controls_available"] = ""
dataset_b["risk_label"] = ""

# Reorder columns to match professor's exact schema order
schema_order = [
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

dataset_b = dataset_b[schema_order]

# Save incomplete dataset
dataset_b.to_csv(OUTPUT_PATH, index=False)

print(f"Incomplete Dataset B saved successfully to: {OUTPUT_PATH}")
print(f"Shape: {dataset_b.shape}")
print("\nFirst row preview:")
print(dataset_b.iloc[0].to_dict())