# Robust Face Recognition: Baseline and Fine-Tuned Performance Analysis

## Overview

This project demonstrates evaluation and fine-tuning of a deep face recognition model on low-quality real-world images.

The goal was to:

- Extract discriminative face embeddings
- Perform strict train/test separation
- Evaluate baseline performance
- Fine-tune model components
- Re-evaluate performance
- Ensure zero data leakage

This project simulates a production-grade AI evaluation workflow.

---

## Features

- Strict dataset split (no identity leakage)
- Cosine similarity-based verification
- ROC curve analysis
- AUC computation
- Optimal threshold selection (Youden's J statistic)
- Similarity score distribution visualization
- Interactive Streamlit evaluation dashboard

---

## Project Structure

- `app/` – Streamlit evaluation dashboard
- `data/` – Embeddings and label files
- `models/` – Trained models and projections
- `notebooks/` – Model training and experimentation

---

## Evaluation Methodology

### 1. Baseline Evaluation
- Extract embeddings from pretrained model
- Normalize embeddings
- Compute pairwise cosine similarity
- Generate ROC curve
- Compute AUC score
- Analyze similarity distributions

### 2. Data Leakage Verification
Strict identity-based separation between training and evaluation sets.

Leakage detection logic ensures:
train_ids ∩ test_ids = ∅


### 3. Metrics
- AUC Score
- ROC Curve
- Optimal Similarity Threshold
- Same vs Different Identity Distribution

---

## Streamlit Dashboard

To run the evaluation dashboard:

```bash
streamlit run app/streamlit_app.py
