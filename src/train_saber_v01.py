"""

==========================================
SABER v0.1 Training Script
==========================================

Algorithm:
    Logistic Regression

Vectorizer:
    TF-IDF

Configuration:
    max_features = 30000
    sublinear_tf = True

Purpose:
    Train the SABER depression detection model and save the trained model along with the TF-IDF vectorizer.

Author:
    Piyush Chaurasiya

"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

print("Loading dataset...")

df = pd.read_csv(
    "data/saber_mini_dataset.csv"
)

print("Dataset Shape:", df.shape)

X = df["text"]
y = df["label"]

# -----------------------------
# Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    sublinear_tf=True
    
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print("Training SABER v0.1...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Making predictions...")

predictions = model.predict(
    X_test_tfidf
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# Save Model
joblib.dump(
    model,
    "models/saber_v01.pkl"
)

# Save Vectorizer
joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully!")