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

  $$I_f = \sum_{t=1}^{15} \left| \phi_{t,f} \right|, \quad I \in \mathbb{R}^6$$

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
```

---

## Installation & Setup

### 1. Prerequisites

- **Python:** 3.9 or higher
- **Package manager:** `pip`

### 2. Install Dependencies

Create a virtual environment (optional) and install the required packages:

```bash
pip install -r requirements.txt
```

Or install the dependencies directly:

```bash
pip install streamlit torch numpy pandas shap matplotlib seaborn scikit-learn
```

### 3. Launch the Application

Run the Streamlit server from the root directory containing `app.py` and the `.pth` weight files:

```bash
streamlit run app.py
```

The application opens automatically in your browser at **http://localhost:8501**.

---

## User Guide: How to Operate the Dashboard

### Step 1: Configure Target Hardware, Architecture & Consensus Strategy (Left Sidebar)

- **Target Hardware Vendor:** Select one of the three supported enterprise drive models:
  - Seagate `ST4000DM000`
  - Toshiba `MG07ACA14TA`
  - HGST `HUH721212ALN604`
- **Primary Architecture (XAI & Detailed Metrics):** Choose **BiLSTM**, **GRU**, or **CNN1D** to control which model drives the SHAP sensor attribution plot in Tab 1 and the detailed single-model report in Tab 3.
- **Consensus Aggregation Strategy:** Select how the three architectures combine their predictions during live diagnostics:
  - `Recall-Weighted Soft Voting`
  - `Unweighted Mean Probability`
  - `Majority Hard Voting (2/3)`
- **Loaded Model Weights:** Verify that all three green status indicators (`BiLSTM`, `GRU`, `CNN1D`) confirm active weight loading in the sidebar.

### Step 2: Run Multi-Model Diagnostics (Tab 1: Live Diagnostics)

Evaluate drive health using either **uploaded CSV telemetry** or the **synthetic signal simulator**.

#### Option A: Upload Telemetry CSV (15-Day Sequence)

1. Select **Upload Telemetry CSV (15-Day Sequence)**.
2. **To test immediately:** Click **Download Sample Healthy CSV** or **Download Sample Failing CSV** to download a 15-day template matched to your selected vendor, then drag and drop it into the uploader.
3. **To upload custom or raw Backblaze CSV logs:** The CSV must contain **at least 15 valid chronological rows**. If longer histories are uploaded, the engine automatically sorts by `date` (if present), applies forward-fill (`ffill()`) imputation, drops leading `NaN` rows, and slices the **most recent 15 days**. Both friendly headers and raw Backblaze column names are supported:

   | Hardware Vendor | Accepted Friendly Headers | Accepted Raw Backblaze Headers |
   | --- | --- | --- |
   | **Seagate ST4000DM000** | `SMART 5`, `SMART 9`, `SMART 187`, `SMART 188`, `SMART 197`, `SMART 198` | `smart_5_raw`, `smart_9_raw`, `smart_187_raw`, `smart_188_raw`, `smart_197_raw`, `smart_198_raw` |
   | **Toshiba MG07ACA14TA** | `SMART 5`, `SMART 9`, `SMART 192`, `SMART 193`, `SMART 197`, `SMART 198` | `smart_5_raw`, `smart_9_raw`, `smart_192_raw`, `smart_193_raw`, `smart_197_raw`, `smart_198_raw` |
   | **HGST HUH721212ALN604** | `SMART 5`, `SMART 9`, `SMART 192`, `SMART 193`, `SMART 197`, `SMART 198` | `smart_5_raw`, `smart_9_raw`, `smart_192_raw`, `smart_193_raw`, `smart_197_raw`, `smart_198_raw` |

4. **Apply Z-Score Standardization toggle:**
   - **Leave unchecked** when uploading pre-standardized tensors, the built-in sample CSV downloads, or sequences exported from `X_tensor*.npy`.
   - **Check the box** when uploading unnormalized raw S.M.A.R.T. counts so `StandardScaler` normalizes the sequence prior to inference.
5. Click **Execute Multi-Model OS Diagnostics**.

#### Option B: Synthetic Signal Simulator

1. Select **Synthetic Signal Simulator**.
2. Choose either **Simulate Healthy Drive** (baseline Gaussian variance, $\mathcal{N}(0, 0.5^2)$) or **Simulate Failing Drive** (injects a $+3.5\sigma$ degradation surge across Days 11–15).
3. Click **Execute Multi-Model OS Diagnostics**.

#### Interpreting Multi-Model Diagnostic Outputs

| Output Component | Meaning |
| --- | --- |
| **4-Card Probability Row** | Displays individual Sigmoid degradation probabilities, `HEALTHY`/`FAILING` badges, CPU inference latency (`ms`), and validation Recall weights for **Bi-LSTM**, **Bi-GRU**, **1D-CNN**, and the **Aggregated Consensus**. |
| **Consensus Status Banner** | Red **CRITICAL ALARM: HARDWARE FAILURE PREDICTED BY CONSENSUS** or Green **DRIVE HEALTH OPTIMAL BY CONSENSUS** based on the active voting strategy. |
| **Method Comparison Table** | Side-by-side table comparing `Bi-LSTM`, `Bi-GRU`, `1D-CNN`, `Unweighted Mean Aggregation`, and `Recall-Weighted Aggregation`. |
| **XAI Sensor Attribution** | Horizontal bar chart of cumulative 15-day SHAP values across the 6 S.M.A.R.T. registers for the primary selected architecture. |

### Step 3: Explore Dataset Demographics (Tab 2: Dataset Analytics)

- View total compiled 15-day sequences, healthy sequences, and verified hardware failures for the selected vendor.
- Inspect the vendor-specific feature engineering table mapping each isolated S.M.A.R.T. attribute to its physical degradation mechanism.

### Step 4: Audit Model Evaluation Metrics (Tab 3: Model Performance)

- **Row 1 (Standard Classification Metrics):** Displays **Precision**, **F1-Score**, **Accuracy**, and **Recall (Sensitivity)** first.
- **Row 2 (Diagnostic & Asymmetric Minority-Class Weighted Cross-Entropy Metrics):** Displays **Balanced Accuracy**, **Specificity (TNR)**, **False Alarm Rate (FAR)** against the `< 0.1%` industry target, and **Asymmetric Weight (`pos_weight`)**.
- **Detailed Diagnostic Report:** Displays the 2×2 **Confusion Matrix** (`TN, FP, FN, TP`) alongside the full **Classification Report** (`Class 0.0`, `Class 1.0`, `Accuracy`, `Macro Avg`, `Weighted Avg`).
- **Master 9-Model Cross-Vendor Benchmark:** A unified 11-column benchmark table comparing all 9 hardware-architecture models.

---

## Master 9-Model Benchmark Summary

All models were trained on an **NVIDIA A100-SXM4-40GB GPU** using an 80/20 train-validation split and Early Stopping. Best performance per vendor is highlighted in **bold**.

| Hardware Vendor (Dataset) | Architecture | Precision | F1-Score | Accuracy (%) | Recall (%) | Balanced Acc (%) | Specificity (%) | FAR (%) | Macro Precision | Macro F1 | Confusion Matrix (TN, FP, FN, TP) | Stop Epoch / A100 Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Seagate ST4000DM000** | **Bi-LSTM** | **0.9999** | **0.9998** | **99.9674** | **100.00** | **99.98** | **99.9674** | **0.0326** | **0.5982** | **0.6641** | 275632, 90, 0, 22 | Ep 25 / 6.50m |
| Seagate ST4000DM000 | Bi-GRU | 0.9999 | 0.9995 | 99.9075 | 100.00 | 99.95 | 99.9075 | 0.0925 | 0.5397 | 0.5733 | 275467, 255, 0, 22 | Ep 28 / 5.79m |
| Seagate ST4000DM000 | 1D-CNN | 0.9999 | 0.9996 | 99.9362 | 86.36 | 93.15 | 99.9373 | 0.0627 | 0.5495 | 0.5886 | 275549, 173, 3, 19 | Ep 24 / 4.34m |
| **Toshiba MG07ACA14TA** | **Bi-LSTM** | 0.9999 | 0.9994 | 99.9007 | **78.72** | **89.31** | 99.9025 | 0.0975 | 0.5308 | 0.5569 | 576637, 563, 10, 37 | Ep 25 / 10.80m |
| **Toshiba MG07ACA14TA** | **Bi-GRU** | **0.9999** | **0.9996** | **99.9298** | 74.47 | 87.20 | **99.9319** | **0.0681** | **0.5409** | **0.5735** | 576807, 393, 12, 35 | Ep 22 / 9.19m |
| Toshiba MG07ACA14TA | 1D-CNN | 0.9999 | 0.9994 | 99.8992 | 42.55 | 71.23 | 99.9038 | 0.0962 | 0.5174 | 0.5319 | 576645, 555, 27, 20 | Ep 13 / 4.88m |
| HGST HUH721212ALN604 | Bi-LSTM | 1.0000 | 0.9997 | 99.9431 | 100.00 | 99.97 | 99.9431 | 0.0569 | 0.5208 | 0.5399 | 161573, 92, 0, 4 | Ep 23 / 2.75m |
| **HGST HUH721212ALN604** | **Bi-GRU** | **1.0000** | **0.9998** | **99.9629** | **100.00** | **99.98** | **99.9629** | **0.0371** | **0.5312** | **0.5587** | 161605, 60, 0, 4 | Ep 40 / 4.73m |
| HGST HUH721212ALN604 | 1D-CNN | 1.0000 | 0.9994 | 99.8856 | 100.00 | 99.94 | 99.8856 | 0.1144 | 0.5106 | 0.5204 | 161480, 185, 0, 4 | Ep 27 / 3.03m |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Sidebar shows a `.pth` weight file is missing | Ensure all 9 `.pth` files are located in the same root directory as `app.py` with the exact filenames listed in the Repository Structure. |
| CSV upload is rejected | Verify the CSV contains at least 15 valid rows and includes the 6 required S.M.A.R.T. columns for the selected vendor (either friendly or raw Backblaze headers). |
| Predictions appear miscalibrated on a raw CSV | Check **Apply Z-Score Standardization** when uploading raw, unnormalized S.M.A.R.T. integer counts; leave it unchecked for pre-standardized tensors and the built-in sample CSVs. |
| Port 8501 is already in use | Run `streamlit run app.py --server.port 8502`. |

---

## Deploying to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and the 9 `.pth` weight files to your GitHub repository. (**Do not push** the multi-gigabyte raw CSVs or `.npy` tensor files in `preprocess_scripts/`).
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and click **Create app**.
3. Select your GitHub repository, branch (`main`), and set `app.py` as the main file path.
4. Click **Deploy**.

---

## Authors

- **Tejas Khare** — School of Computer Science and Engineering, Vellore Institute of Technology ([tejas.khare2024@vitstudent.ac.in](mailto:tejas.khare2024@vitstudent.ac.in))
- **Yash Lohiya** — School of Computer Science and Engineering, Vellore Institute of Technology ([yash.lohiya2024@vitstudent.ac.in](mailto:yash.lohiya2024@vitstudent.ac.in))
