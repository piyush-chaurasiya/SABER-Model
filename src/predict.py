"""
==========================================
SABER Prediction Script
==========================================

Purpose:
Load the trained SABER model and predict
whether the given text indicates depression.

Model:
Logistic Regression + TF-IDF

Author:
Piyush Chaurasiya
"""

import joblib


# -----------------------------
# Load Trained Model
# -----------------------------

model = joblib.load(
    "models/saber_v01.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# -----------------------------
# Welcome Banner
# -----------------------------

print("=" * 50)
print("               SABER v0.1")
print("      AI Depression Detection System")
print("=" * 50)

print("\nType 'exit' anytime to quit.\n")


# -----------------------------
# Prediction Loop
# -----------------------------

while True:

    text = input("Enter Text: ")

    # Exit Program
    if text.lower() == "exit":
        print("\nThank you for using SABER.")
        print("Goodbye!\n")
        break

    # Empty Input Check
    if text.strip() == "":
        print("\nPlease enter some text.\n")
        continue

    # Convert Text into TF-IDF Features
    text_tfidf = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_tfidf)[0]

    # Prediction Probability
    probability = model.predict_proba(text_tfidf)[0]

    not_depressed = probability[0] * 100
    depressed = probability[1] * 100

    # Risk Level
    if depressed >= 80:
        risk = "HIGH"

    elif depressed >= 50:
        risk = "MODERATE"

    else:
        risk = "LOW"

    # Display Result
    print("\n" + "=" * 50)
    print("Prediction Result")
    print("-" * 50)

    if prediction == 1:
        print("Prediction              : DEPRESSED")
    else:
        print("Prediction              : NOT DEPRESSED")

    print(f"Confidence (Depressed)  : {depressed:.2f}%")
    print(f"Confidence (Normal)     : {not_depressed:.2f}%")
    print(f"Risk Level              : {risk}")

    print("=" * 50)