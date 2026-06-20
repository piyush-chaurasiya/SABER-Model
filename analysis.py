import pandas as pd

df = pd.read_csv(
    "data/reddit_depression_dataset.csv",
    low_memory=False
)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\n========== SAMPLE DEPRESSED POST ==========")

depressed = df[df["label"] == 1].iloc[0]

print("TITLE:")
print(depressed["title"])

print("\nBODY:")
print(depressed["body"])

print("\n========== SAMPLE NON-DEPRESSED POST ==========")

normal = df[df["label"] == 0].iloc[0]

print("TITLE:")
print(normal["title"])

print("\nBODY:")
print(normal["body"])

print("\n\n===== 5 RANDOM DEPRESSED POSTS =====")

depressed_samples = df[df["label"] == 1].sample(5, random_state=42)

for i, row in depressed_samples.iterrows():
    print("\n------------------")
    print("TITLE:", row["title"])
    print("BODY:", row["body"])

print("\n\n===== 5 RANDOM NON-DEPRESSED POSTS =====")

normal_samples = df[df["label"] == 0].sample(5, random_state=42)

for i, row in normal_samples.iterrows():
    print("\n------------------")
    print("TITLE:", row["title"])
    print("BODY:", row["body"])


print("\n\nMissing Values:\n")
print(df[["title", "body"]].isnull().sum())