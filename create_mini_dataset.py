import pandas as pd

print("Loading dataset...")

df = pd.read_csv(
    "data/saber_dataset_v1.csv"
)

print("Original Shape:", df.shape)

# Separate classes
depressed = df[df["label"] == 1]
normal = df[df["label"] == 0]

# Random sample
depressed_sample = depressed.sample(
    n=50000,
    random_state=42
)

normal_sample = normal.sample(
    n=50000,
    random_state=42
)

# Combine
mini_df = pd.concat(
    [depressed_sample, normal_sample]
)

# Shuffle
mini_df = mini_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("Mini Dataset Shape:")
print(mini_df.shape)

print("\nLabel Distribution:")
print(mini_df["label"].value_counts())

mini_df.to_csv(
    "data/saber_mini_dataset.csv",
    index=False
)

print("\nMini Dataset Saved!")