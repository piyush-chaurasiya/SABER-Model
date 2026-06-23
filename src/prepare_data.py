import pandas as pd

print("Loading dataset...")

df = pd.read_csv(
    "data/reddit_depression_dataset.csv",
    low_memory=False
)

# Missing values handle
df["title"] = df["title"].fillna("")
df["body"] = df["body"].fillna("")

# Final text column
df["text"] = (
    df["title"].astype(str)
    + " "
    + df["body"].astype(str)
)

# Keep only useful columns
df = df[["text", "label"]]

print(df.head())

print("\nShape:")
print(df.shape)

# Save
df.to_csv(
    "data/saber_dataset_v1.csv",
    index=False
)

print("\nDataset Saved Successfully!")