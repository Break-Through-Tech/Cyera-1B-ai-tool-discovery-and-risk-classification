# PURPOSE:
#
# Analyze the cleaned Kaggle data
# before formatting it as AI Tool Metadata

import pandas as pd
import numpy as np

DATA_PATH = "data/cleaned_kaggle.csv"

df = pd.read_csv(DATA_PATH)

if "Website" in df.columns:
    formatted_website = df["Website"].str.startswith("https://", na = False) & df["Website"].str.endswith("?ref=aitoolbuzz.com")

    count = formatted_website.sum()
    print(f"Correctly formatted websites: {count}")
    print(f"Total websites: {df.shape[0]}")

    incorrect_format = df[~formatted_website]

    print(f"Incorrectly formatted websites: {incorrect_format.shape[0]}")
    print("\nSample of incorrect rows:")
    print(incorrect_format[["Website"]])