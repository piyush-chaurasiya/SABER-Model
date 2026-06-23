import joblib

# Load model
model = joblib.load(
    "models/saber_v01.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

print("SABER v0.1 Loaded")

while True:

    text = input("\nEnter Text: ")

    if text.lower() == "exit":
        break

    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]

    if prediction == 1:
        print("Prediction: DEPRESSED")
    else:
        print("Prediction: NOT DEPRESSED")

    probability = model.predict_proba(text_tfidf)[0]

    not_depressed = probability[0] * 100

    depressed = probability[1] * 100

    print(f"Not Depressed: {not_depressed:.2f}%")

    print(f"Depressed: {depressed:.2f}%")