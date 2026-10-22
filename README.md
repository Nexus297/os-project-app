# Universal S.M.A.R.T. Degradation Probability Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)

An Operating System-level predictive maintenance and Explainable AI (XAI) diagnostic web application built with **Streamlit**, **PyTorch**, and **SHAP**.

The engine evaluates **15-day sequential S.M.A.R.T.** (Self-Monitoring, Analysis, and Reporting Technology) telemetry to predict enterprise Hard Disk Drive (HDD) failures across multiple hardware vendors using **Asymmetric Minority-Class Weighted Cross-Entropy (`BCEWithLogitsLoss`)** and **Multi-Model Consensus Voting**, **without relying on synthetic oversampling (e.g., SMOTE)**.

---

## Table of Contents

- [Key Features](#key-features)
- [Dataset Overview](#dataset-overview)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [User Guide: How to Operate the Dashboard](#user-guide-how-to-operate-the-dashboard)
- [Master 9-Model Benchmark Summary](#master-9-model-benchmark-summary)
- [Troubleshooting](#troubleshooting)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Authors](#authors)

---

## Key Features

- **Cross-Vendor Hardware Support:** Dynamically maps vendor-specific S.M.A.R.T. firmware registers across **Seagate (`ST4000DM000`)**, **Toshiba (`MG07ACA14TA`)**, and **HGST (`HUH721212ALN604`)**.
- **Parallel Multi-Architecture Inference & Consensus Voting:** Simultaneously executes temporal recurrent networks (**Bidirectional LSTM** and **Bidirectional GRU**) alongside a spatial feature extractor (**1D-CNN**) and aggregates their predictions via three selectable consensus strategies:
  1. **Recall-Weighted Soft Voting:** Weights each model's probability by its empirical validation Recall on the target vendor.
  2. **Unweighted Mean Probability:** Computes the arithmetic mean across all three architectures.
  3. **Majority Hard Voting (2/3):** Triggers an OS critical alarm if at least 2 out of 3 architectures flag the sequence as failing ($p_a > 0.50$).
- **Asymmetric Minority-Class Weighted Cross-Entropy:** Eliminates synthetic time-series distortion by applying exact dynamic positive-class penalty weights ($W_p = N_{\text{Healthy}} / N_{\text{Failed}}$) inside `BCEWithLogitsLoss`.
- **Live 15-Day CSV Telemetry Ingestion:** Accepts raw or pre-standardized `.csv` drive logs, automatically handles chronological sorting and forward-fill (`ffill`) imputation, optionally applies Z-score standardization (`StandardScaler`), and slices the most recent 15-day window (shape `1 × 15 × 6`).
- **Built-In Sample CSV Generator & Synthetic Simulator:** One-click downloadable healthy and failing CSV templates per vendor, plus an interactive synthetic anomaly generator.
- **Explainable AI (XAI) Root-Cause Attribution:** Uses `shap.GradientExplainer` over a 100-sample baseline reference tensor to compute cumulative 15-day sensor importance, isolating the exact physical S.M.A.R.T. attribute driving the prediction:

  $$I_f = \sum_{t=1}^{15} \left\vert{}\phi_{t,f}\right\vert{} \in \mathbb{R}^6$$

- **OS Daemon Latency Profiling:** Measures per-model (**< 5 ms**) and parallel multi-model (**< 12 ms**) CPU inference execution time in milliseconds.
- **Complete Standard & Diagnostic Evaluation Suite:** Prioritizes standard global classification metrics (**Precision**, **F1-Score**, **Accuracy**, and **Recall**) alongside domain-specific diagnostic indicators (**Balanced Accuracy**, **Specificity / TNR**, **False Alarm Rate / FAR**, **Asymmetric Penalty Weight**, **Macro Precision**, **Macro F1**, **2×2 Confusion Matrices**, and a **Master 9-Model Benchmark Table**).

---

## Dataset Overview

The models are trained and evaluated on the **Backblaze Q1 2023** enterprise drive-stats archive (91 daily CSV snapshots across 90+ S.M.A.R.T. attributes).

| Property | Value |
| --- | --- |
| Healthy drive-day logs (global) | **99.99567%** |
| Failure drive-day logs (global) | **0.00433%** |
| Total compiled 15-day sequences | **5,073,294** (`1,378,717` Seagate, `2,886,234` Toshiba, `808,343` HGST) |
| Input window | 15 consecutive days (`stride = 1`) |
| Features per day | 6 isolated S.M.A.R.T. attributes per vendor |
| Input tensor shape | `(1, 15, 6)` |

Because of the extreme class imbalance (`9,573.4x` on Seagate, `33,174.1x` on Toshiba, and `11,714.1x` on HGST), the models utilize **Asymmetric Minority-Class Weighted Cross-Entropy** (`pos_weight` in `BCEWithLogitsLoss`) instead of synthetic oversampling.

---

## Repository Structure

Ensure `app.py`, `requirements.txt`, and all **9 pre-trained PyTorch weight files** (`.pth`) reside in the **root directory**:

```text
├── app.py                                      # Main multi-model Streamlit diagnostic application
├── os_project_1.ipynb                          # NVIDIA A100 training & evaluation notebook
├── requirements.txt                            # Python package dependencies
├── README.md                                   # Project documentation
│
├── Pre-Trained Model Weights (root directory):
│   ├── bilstm_hdd_best.pth                     # Seagate ST4000DM000 - Bi-LSTM weights
│   ├── gru_ST4000DM000_best.pth                # Seagate ST4000DM000 - Bi-GRU weights
│   ├── cnn1d_ST4000DM000_best.pth              # Seagate ST4000DM000 - 1D-CNN weights
│   ├── bilstm_TOSHIBA_MG07ACA14TA_best.pth     # Toshiba MG07ACA14TA - Bi-LSTM weights
│   ├── gru_TOSHIBA_MG07ACA14TA_best.pth        # Toshiba MG07ACA14TA - Bi-GRU weights
│   ├── cnn1d_TOSHIBA_MG07ACA14TA_best.pth      # Toshiba MG07ACA14TA - 1D-CNN weights
│   ├── bilstm_HGST_HUH721212ALN604_best.pth    # HGST HUH721212ALN604 - Bi-LSTM weights
│   ├── gru_HGST_HUH721212ALN604_best.pth       # HGST HUH721212ALN604 - Bi-GRU weights
│   └── cnn1d_HGST_HUH721212ALN604_best.pth     # HGST HUH721212ALN604 - 1D-CNN weights
│
├── eda/
│   ├── module_1_eda.py                         # Global class imbalance & correlation audit
│   └── generate_advanced_eda.py                # Cross-vendor correlation, trajectory & benchmark plots
│
├── plots/
│   ├── eda_class_imbalance.png                 # Global healthy vs. failure distribution pie chart
│   ├── eda_failure_distribution.png            # Top 10 failing drive models in Q1 2023
│   ├── eda_feature_correlation.png             # Baseline S.M.A.R.T. Pearson correlation matrix
│   ├── eda_vendor_correlation_grid.png         # 3-panel vendor-isolated correlation heatmaps
│   ├── eda_temporal_degradation_trajectory.png # 15-day Z-score degradation trajectory curves
│   ├── eda_sensor_boxplot_distributions.png    # Day-15 healthy vs. failed sensor boxplots
│   ├── eda_cross_vendor_imbalance_weights.png  # Sequence volume vs. asymmetric penalty weight (Wp)
│   └── results_9model_benchmark.png            # 9-model Recall (%) and FAR (%) benchmark chart
│
└── preprocess_scripts/
    ├── model_inventory_scanner.py              # Hardware prefix & failure density scanner
    ├── seagate_2d_imputation.py                # Chronological sorting & drive-grouped ffill()
    ├── seagate_3d_tensor_compiler.py           # Per-drive Z-score scaling & 15-day windowing
    ├── multivendor_tensor_pipeline.py          # Automated Toshiba & HGST 3D tensor compiler
    └── *.npy                                   # Compiled 3D feature & label tensors (local only)
