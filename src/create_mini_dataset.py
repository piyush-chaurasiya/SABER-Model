"""
==========================================
SABER Mini Dataset Creation Script
==========================================

Purpose:
Create a balanced mini dataset from the
processed SABER dataset for faster model
training and experimentation.

Input:
    data/saber_dataset_v1.csv

Output:
    data/saber_mini_dataset.csv

Author:
    Piyush Chaurasiya
"""

import pandas as pd


# -----------------------------
# Load Processed Dataset
# -----------------------------

print("Loading dataset...")

df = pd.read_csv(
    "data/saber_dataset_v1.csv"
)

print(f"Original Dataset Shape: {df.shape}")


# -----------------------------
# Separate Classes
# -----------------------------

depressed = df[df["label"] == 1]

normal = df[df["label"] == 0]


# -----------------------------
# Random Sampling
# -----------------------------

depressed_sample = depressed.sample(
    n=50000,
    random_state=42
)

normal_sample = normal.sample(
    n=50000,
    random_state=42
)


# -----------------------------
# Create Balanced Dataset
# -----------------------------

mini_df = pd.concat(
    [depressed_sample, normal_sample]
)


# -----------------------------
# Shuffle Dataset
# -----------------------------

mini_df = mini_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# -----------------------------
# Display Dataset Information
# -----------------------------

print(f"\nMini Dataset Shape: {mini_df.shape}")

print("\nLabel Distribution:")

print(mini_df["label"].value_counts())


# -----------------------------
# Save Mini Dataset
# -----------------------------

mini_df.to_csv(
    "data/saber_mini_dataset.csv",
    index=False
)


# -----------------------------
# Success Message
# -----------------------------

print("\nBalanced mini dataset created successfully!")
print("Location: data/saber_mini_dataset.csv")