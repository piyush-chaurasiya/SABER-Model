# SABER Experiment Results

This document records all machine learning experiments performed during the development of the **SABER (Social Media and Affective Behavior Recognition)** model.

---

# Experiment Summary

| Version | Algorithm | Configuration | Accuracy | Status |
|---------|-----------|---------------|---------:|--------|
| v0.1 | Logistic Regression | TF-IDF (max_features=10000) | 92.33% | Baseline |
| v0.2 | Linear SVM | TF-IDF (max_features=10000) | 92.12% | Tested |
| v0.3 | Multinomial Naive Bayes | TF-IDF (max_features=10000) | 88.57% | Tested |
| v0.4 | Random Forest | TF-IDF (max_features=10000) | 87.83% | Tested |
| v0.5 | Logistic Regression | TF-IDF (max_features=30000) | 92.37% | Improved |
| Final | Logistic Regression | TF-IDF (max_features=30000, sublinear_tf=True) | **92.42%** | ✅ Selected |

---

# Experiment Details

---

## SABER v0.1

### Algorithm

Logistic Regression

### Configuration

- TF-IDF
- max_features = 10000

### Accuracy

92.33%

### Observation

Established the first baseline model with strong overall performance.

---

## SABER v0.2

### Algorithm

Linear Support Vector Machine (Linear SVM)

### Configuration

- TF-IDF
- max_features = 10000

### Accuracy

92.12%

### Observation

Performance was very close to Logistic Regression but slightly lower.

---

## SABER v0.3

### Algorithm

Multinomial Naive Bayes

### Configuration

- TF-IDF
- max_features = 10000

### Accuracy

88.57%

### Observation

Achieved high recall for depressed posts but lower overall accuracy and precision.

---

## SABER v0.4

### Algorithm

Random Forest

### Configuration

- TF-IDF
- max_features = 10000

### Accuracy

87.83%

### Observation

Training time increased significantly while accuracy decreased. Random Forest was not suitable for high-dimensional sparse TF-IDF features.

---

## SABER v0.5

### Algorithm

Logistic Regression

### Configuration

- TF-IDF
- max_features = 30000

### Accuracy

92.37%

### Observation

Increasing the vocabulary size provided a small improvement over the baseline model.

---

## Final SABER Classical Model

### Algorithm

Logistic Regression

### Configuration

- TF-IDF
- max_features = 30000
- sublinear_tf = True

### Accuracy

**92.42%**

### Observation

This configuration achieved the best overall performance among all classical machine learning experiments.

---

# Final Model

Algorithm

- Logistic Regression

Vectorizer

- TF-IDF

Configuration

- max_features = 30000
- sublinear_tf = True

Accuracy

**92.42%**

Model File

```
models/saber_v01.pkl
```

Vectorizer

```
models/tfidf_vectorizer.pkl
```

---

# Conclusion

A total of four classical machine learning algorithms and multiple TF-IDF configurations were evaluated.

Among all experiments, **Logistic Regression combined with an optimized TF-IDF vectorizer (`max_features=30000`, `sublinear_tf=True`)** produced the best overall performance.

This model has been selected as the baseline depression detection engine for the PersonaSense project and will be used for future comparisons with transformer-based models such as DistilBERT.