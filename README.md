# PersonaSense

### AI-Based User Behavior Intelligence System

**PersonaSense** is an AI-powered user behavior intelligence system designed to identify signs of depression through textual analysis of social media content.

The project aims to assist in the early detection of depression by analyzing user-generated text using Machine Learning and, in future versions, Deep Learning models.

---

# AI Engine

## SABER

**SABER** (Social Media and Affective Behavior Recognition)

SABER is the machine learning engine of PersonaSense responsible for detecting depression from textual data collected from social media platforms.

---

# Project Objective

The primary objective of PersonaSense is to develop an intelligent framework capable of identifying potential depression risks through social media activity and behavioral analysis.

The system is designed to:

- Detect signs of depression from text.
- Analyze behavioral patterns.
- Estimate depression risk.
- Support early mental health awareness.
- Serve as the AI engine for the PersonaSense platform.

---

# Dataset

Two datasets are planned for this project.

## 1. Reddit Depression Dataset

Used for depression detection.

Original Dataset

```
reddit_depression_dataset.csv
```

Shape

```
(2470778, 8)
```

Columns

```
Unnamed: 0
subreddit
title
body
upvotes
created_utc
num_comments
label
```

Label Distribution

```
Not Depressed (0) : 1,990,261

Depressed (1)     : 480,411
```

---

## Processed Dataset

```
saber_dataset_v1.csv
```

Shape

```
(2470778, 2)
```

Columns

```
text
label
```

---

## Mini Dataset

```
saber_mini_dataset.csv
```

Shape

```
(100000, 2)
```

Label Distribution

```
Not Depressed : 50000

Depressed     : 50000
```

The mini dataset is balanced and is currently used for training and evaluating SABER Version 1.

---

# Current Progress

✅ Dataset Analysis

✅ Data Preparation

✅ Mini Dataset Creation

✅ Logistic Regression

✅ Linear SVM

✅ Naive Bayes

✅ Random Forest

✅ TF-IDF Optimization

✅ SABER v0.1 (Best Classical ML Model)

---

Author

**Piyush Chaurasiya**

B.Tech CSE

Greater Noida Institute of Technology