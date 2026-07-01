"""
==========================================
SABER Data Preparation Script
==========================================

Purpose:
Clean the raw Reddit depression dataset,
combine the title and body into a single
text column, and save the processed dataset.

Input:
    data/reddit_depression_dataset.csv

Output:
    data/saber_dataset_v1.csv

Author:
    Piyush Chaurasiya
"""

import pandas as pd


# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(
    "data/reddit_depression_dataset.csv",
    low_memory=False
)


# -----------------------------
# Handle Missing Values
# -----------------------------

df["title"] = df["title"].fillna("")
df["body"] = df["body"].fillna("")


# -----------------------------
# Create Final Text Column
# -----------------------------

df["text"] = (
    df["title"].astype(str)
    + " "
    + df["body"].astype(str)
)


# -----------------------------
# Keep Required Columns
# -----------------------------

df = df[["text", "label"]]


# -----------------------------
# Display Dataset Information
# -----------------------------

print("\nFirst 5 Rows:")
print(df.head())

print(f"\nDataset Shape: {df.shape}")


# -----------------------------
# Save Processed Dataset
# -----------------------------

df.to_csv(
    "data/saber_dataset_v1.csv",
    index=False
)


# -----------------------------
# Success Message
# -----------------------------

print("\nProcessed dataset saved successfully!")
print("Location: data/saber_dataset_v1.csv")