# PURPOSE:
#
# Remove tools without an "Active" website
# or without a "Short Description"
# 
# Keep everything else because there is 
# too much missing data. Removing would
# significantly reduce our data quantity.

import pandas as pd
import numpy as np

DATA_PATH = (
    "data/Complete AI Tools Dataset 2025 - 16763 Tools from AIToolBuzz.csv"
)
OUTPUT_PATH = "data/cleaned_kaggle.csv"

df = pd.read_csv(DATA_PATH)

# Basic information
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Data types and non-null counts
print("\nDataFrame info:")
df.info()

# Number of missing values in each column
print("\nMissing values:")
print(df.isna().sum())

# Percentage of missing values in each column
print("\nMissing percentage:")
print((df.isna().mean() * 100).sort_values(ascending=False))

# Basic statistics for numerical columns
print("\nNumerical statistics:")
print(df.describe())

# Strip whitespace from columns
df.columns = df.columns.str.strip()

# Only keep active websites
if "Website Status" in df.columns:
    df = df[
        df["Website Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("active")
    ].copy()

# Clean Short Description + remove empty rows
if "Short Description" in df.columns:
    df["Short Description"] = (
        df["Short Description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df = df[df["Short Description"] != ""].copy()

# Text columns (exluding Website Status and Short Description)
text_cols = [
    "Name",
    "Link",
    "Logo",
    "Category",
    "Primary Task",
    "Keywords",
    "Country",
    "industry",
    "technologies",
    "Website",
]

# Fill NaNs and missing entries in text columns with ""
for col in text_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

# Convert Year Founded to numeric
if "Year Founded" in df.columns:
    df["Year Founded"] = pd.to_numeric(df["Year Founded"], errors="coerce")

# Drop tools that have the same name or website (duplicates)
df = df.drop_duplicates(subset=["Name", "Website"], keep="first")

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nCleaned dataset successfully saved to: {OUTPUT_PATH}")
print(f"Final cleaned shape: {df.shape}")