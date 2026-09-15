# MOF-Synthesis-NLP-ML-Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/mariaaziz074-web/MOF-Synthesis-NLP-ML-Benchmark)

> **Automated literature text-mining, domain-validated dataset curation, and a comprehensive 13-model machine learning benchmark for Metal-Organic Framework (MOF) synthesis parameters.**

---

## 📌 Overview

Extracting structured reaction parameters from unstructured scientific literature is essential for data-driven materials discovery. This repository provides:
1. **Curated MOF Dataset ($N=54$):** Tabular synthesis parameters mined from 54 peer-reviewed review articles, bounded by rigorous chemical ontology rules.
2. **Unsupervised Chemical Space Exploration:** 2D Principal Component Analysis (PCA) and K-Means clustering resolving hydrothermal vs. solvothermal synthesis regimes.
3. **13-Model Supervised ML Benchmark:** Cross-validated evaluation across 6 algorithmic families (Tree Ensembles, Boosting, Kernel Machines, Linear, Probabilistic, Neural Networks) for synthesis solvent prediction.
4. **Publication-Ready Figures:** 300-DPI visual assets formatted for journal submission and presentations.

---

## 🖼️ Master Overview Figure

<p align="center">
  <img src="figures/Master_MOF_Benchmark_Figure.png" alt="MOF Synthesis ML Benchmark Overview" width="95%">
</p>

*Figure 1: (A) Extraction distributions and descriptor coverage across 54 review papers. (B) Metal–solvent and metal–linker co-occurrence frequency heatmaps. (C) 2D PCA chemical space partitioned via K-Means clustering. (D) Cross-validated accuracy leaderboard across 13 machine learning architectures.*

---

## 📊 Dataset Schema (`data/full_realistic_mof_data.csv`)

| Column Name | Type | Physical Range / Values | Extraction Coverage | Description |
| :--- | :---: | :---: | :---: | :--- |
| `filename` | `str` | Source PDF Identifier | **100.0%** (54/54) | Source literature review reference |
| `metal` | `str` | `Mg`, `Ti`, `Cu`, `Al`, `V`, `Eu`, `Ce` | **100.0%** (54/54) | Primary inorganic metal node |
| `solvent` | `str` | `water`, `ethanol`, `dmf`, `acetone` | **77.8%** (42/54) | Primary reaction medium |
| `linker` | `str` | `BTC`, `BPDC`, `BDC` (Terephthalic acid) | **59.3%** (32/54) | Organic bridging ligand |
| `temperature` | `float` | `60.0 – 400.0 °C` (Mean: 184.2 °C) | **11.1%** (6/54) | Reaction / solvothermal temperature |
| `pH` | `float` | `1.0 – 14.0` (Mean: 3.75) | **11.1%** (6/54) | Synthesis medium acidity/basicity |
| `time_hours` | `float` | `2.0 – 48.0 h` (Median: 4.0 h) | **16.7%** (9/54) | Hydrothermal crystallization duration |

---

## 🤖 13-Model Machine Learning Leaderboard

All models were evaluated using **Stratified 4-Fold Cross-Validation** with strict pipeline isolation (zero data leakage) on the solvent selection task:

| Rank | Model Architecture | Family | CV Accuracy ($\mu \pm \sigma$) | Macro-F1 | Precision | Recall |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Linear SVM** | Support Vector Machine | **$0.739 \pm 0.037$** | **0.337** | 0.325 | 0.354 |
| 🥈 | **Random Forest** | Bagging Ensemble | **$0.714 \pm 0.014$** | 0.278 | 0.238 | 0.333 |
| 🥉 | **Extra Trees** | Randomized Ensembles | **$0.714 \pm 0.014$** | 0.278 | 0.238 | 0.333 |
| 4 | **Logistic Regression** | Linear ($L_2$-Regularized) | **$0.689 \pm 0.052$** | 0.276 | 0.238 | 0.323 |
| 5 | **Ridge Classifier** | Linear ($L_2$-Penalized) | **$0.689 \pm 0.052$** | 0.276 | 0.238 | 0.323 |
| 6 | **Hist Gradient Boosting** | Histogram Boosting | **$0.689 \pm 0.052$** | 0.276 | 0.238 | 0.323 |
| 7 | **Gradient Boosting** | Sequential Boosting | **$0.665 \pm 0.064$** | 0.329 | 0.350 | 0.344 |
| 8 | **Multi-Layer Perceptron** | Neural Network (32×16) | **$0.665 \pm 0.064$** | 0.274 | 0.238 | 0.312 |
| 9 | **AdaBoost** | Adaptive Boosting | **$0.642 \pm 0.076$** | 0.324 | 0.362 | 0.333 |
| 10 | **Decision Tree** | White-box Tree | **$0.617 \pm 0.088$** | 0.268 | 0.238 | 0.292 |
| 11 | **RBF Kernel SVM** | Non-linear Kernel | **$0.617 \pm 0.088$** | 0.268 | 0.238 | 0.292 |
| 12 | **K-Nearest Neighbors** | Instance-Based (k=3) | **$0.573 \pm 0.131$** | 0.255 | 0.238 | 0.281 |
| 13 | **Gaussian Naive Bayes** | Probabilistic | **$0.427 \pm 0.150$** | 0.250 | 0.245 | 0.271 |

---

## 📁 Repository Structure

```text
MOF-Synthesis-NLP-ML-Benchmark/
│
├── README.md                          <-- Project documentation
├── LICENSE                            <-- MIT Open Source License
├── requirements.txt                   <-- Python package dependencies
├── .gitignore                         <-- Excludes raw PDFs and cache
│
├── data/                              <-- Curated datasets
│   ├── full_realistic_mof_data.csv    <-- Master tabular dataset (54 × 7)
│   └── full_realistic_mof_data.xlsx   <-- Excel version
│
├── src/                               <-- Modular execution pipeline
│   ├── 01_extract_descriptors.py      <-- Regex NLP extraction engine
│   ├── 02_verify_dataset.py           <-- Dataset validation & sanity checker
│   ├── 03_run_ml_experiments.py       <-- PCA & 13-model benchmark
│   └── 04_generate_figures.py         <-- 300-DPI figure generator
│
├── results/                           <-- Quantitative metrics (CSV logs)
│   ├── 13_model_leaderboard.csv       <-- Cross-validation metrics
│   ├── solvent_benchmark.csv          <-- Solvent classification details
│   └── metal_feature_importances.csv  <-- Gini importance scores
│
└── figures/                           <-- 300-DPI publication figures
    ├── Master_MOF_Benchmark_Figure.png
    ├── Fig1_Descriptor_Landscape.png
    ├── Fig2_Cooccurrence_Heatmaps.png
    ├── Fig3_PCA_Chemical_Space.png
    ├── Fig4_13_Models_Accuracy_Leaderboard.png
    ├── Fig5_Top_Models_Radar_Comparison.png
    └── Fig6_Confusion_Matrices.png