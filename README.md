# 🧬 RNA Classification using Ensemble Machine Learning

<p align="center">
  <b>Robust RNA sequence classification using feature engineering + gradient boosting ensembles</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-green" />
  <img src="https://img.shields.io/badge/Task-RNA%20Classification-orange" />
  <img src="https://img.shields.io/badge/Metric-ROC--AUC-red" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 📌 Overview

This project implements an end-to-end machine learning pipeline to classify RNA sequences into binary classes (**+1 / -1**).

It combines:
- Biological feature engineering  
- Advanced ensemble learning  
- Robust validation techniques  

to achieve strong predictive performance on unseen data.

---

## 🚀 Key Highlights

- Multi-modal feature extraction:
  - RNA structural features (ViennaRNA)
  - k-mer sequence features
  - Alignment-based similarity (Smith-Waterman, miRBase)

- Ensemble of state-of-the-art models:
  - XGBoost  
  - LightGBM  
  - CatBoost  

- Stratified 5-fold cross-validation  
- Pseudo-labeling for improved generalization  
- Scalable and reproducible pipeline  

---

## 🧠 Methodology

### 🔬 Feature Engineering
RNA sequences are transformed into numerical representations using:
- Structural features: MFE, EFE, base-pairing properties  
- Sequence features: k-mer frequencies  
- Alignment features: similarity scores with miRBase  

---

### ⚙️ Preprocessing
- Missing and infinite values handled  
- Feature scaling using **StandardScaler**  

---

### 🤖 Model Training
- Ensemble of gradient boosting models  
- Stratified 5-fold cross-validation  
- Evaluation metric: **ROC-AUC**  

---

### 📊 Inference
- Predictions averaged across multiple models and seeds  
- Final output generated as probability scores  

---

## 📈 Results

| Model     | ROC-AUC |
|----------|--------|
| XGBoost  | 0.8397 |
| LightGBM | 0.8366 |
| CatBoost | 0.8402 |
| **Ensemble** | **0.8411** |

🏆 **Kaggle Score:** 0.82  

## Note
This project was part of a kaggle competition.
