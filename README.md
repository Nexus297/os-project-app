# Universal S.M.A.R.T. Degradation Probability Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-purple)

An Operating System-level predictive maintenance and Explainable AI (XAI) diagnostic web application built with **Streamlit**, **PyTorch**, and **SHAP**.

The engine evaluates **15-day sequential S.M.A.R.T.** (Self-Monitoring, Analysis, and Reporting Technology) telemetry to predict enterprise Hard Disk Drive (HDD) failures across multiple hardware vendors, **without relying on synthetic oversampling (e.g., SMOTE)**.

---

## Table of Contents

- [Key Features](#key-features)
- [Dataset Overview](#dataset-overview)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [User Guide](#user-guide-how-to-operate-the-dashboard)
- [Master 9-Model Benchmark Summary](#master-9-model-benchmark-summary)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Authors](#authors)

---

## Key Features

- **Cross-Vendor Hardware Support:** Dynamically maps vendor-specific S.M.A.R.T. firmware registers across **Seagate (`ST4000DM000`)**, **Toshiba (`MG07ACA14TA`)**, and **HGST (`HUH721212ALN604`)**.
- **Multi-Architecture Deep Learning:** Benchmarks temporal recurrent networks (**Bidirectional LSTM** and **Bidirectional GRU**) against a spatial feature extractor (**1D-CNN**), all trained with dynamic cost-sensitive `BCEWithLogitsLoss`.
- **Live 15-Day CSV Telemetry Ingestion:** Accepts raw or pre-standardized `.csv` drive logs, automatically handles chronological sorting and forward-fill (`ffill`) imputation, optionally applies Z-score standardization (`StandardScaler`), and slices the most recent 15-day window (shape `1 × 15 × 6`).
- **Built-In Sample CSV Generator & Synthetic Simulator:** One-click downloadable healthy and failing CSV templates per vendor, plus an interactive synthetic anomaly generator.
- **Explainable AI (XAI) Root-Cause Attribution:** Uses `shap.GradientExplainer` to compute cumulative 15-day sensor importance, isolating the exact physical S.M.A.R.T. attribute responsible for triggering an alarm:

  $$I_f = \sum_{t=1}^{15} \left|\phi_{t,f}\right|$$

- **OS Daemon Latency Profiling:** Measures single-sequence CPU inference time in milliseconds (**< 5 ms**).
- **Complete 11-Column Evaluation Suite:** Overall Accuracy, Balanced Accuracy, Failure Recall (Sensitivity), Specificity (TNR), False Alarm Rate (FAR), Failure Precision, Failure F1-Score, Macro F1, Weighted F1, Confusion Matrices, and a Master 9-Model Benchmark Table.

---

## Dataset Overview

The models are trained and evaluated on the **Backblaze Q1 2023** drive-stats dataset.

| Property | Value |
| --- | --- |
| Healthy drive-day logs (global) | **99.99567%** |
| Failure drive-day logs (global) | **0.00433%** |
| Input window | 15 consecutive days |
| Features per day | 6 S.M.A.R.T. attributes |
| Input tensor shape | `(1, 15, 6)` |

Because of this extreme class imbalance, the models use **cost-sensitive learning** (a dynamic `pos_weight` penalty in `BCEWithLogitsLoss`) instead of synthetic oversampling.

---

## Repository Structure

Make sure the project directory contains `app.py`, `requirements.txt`, and all **9 pre-trained PyTorch weight files** (`.pth`) in the **root directory**:

```text
├── app.py                                    # Main Streamlit diagnostic application
├── requirements.txt                          # Python package dependencies
├── README.md                                 # Project documentation
│
├── Model Weights (required in root directory for app.py):
│   ├── bilstm_hdd_best.pth                   # Seagate ST4000DM000 - Bi-LSTM weights
│   ├── gru_ST4000DM000_best.pth              # Seagate ST4000DM000 - Bi-GRU weights
│   ├── cnn1d_ST4000DM000_best.pth            # Seagate ST4000DM000 - 1D-CNN weights
│   ├── bilstm_TOSHIBA_MG07ACA14TA_best.pth   # Toshiba MG07ACA14TA - Bi-LSTM weights
│   ├── gru_TOSHIBA_MG07ACA14TA_best.pth      # Toshiba MG07ACA14TA - Bi-GRU weights
│   ├── cnn1d_TOSHIBA_MG07ACA14TA_best.pth    # Toshiba MG07ACA14TA - 1D-CNN weights
│   ├── bilstm_HGST_HUH721212ALN604_best.pth  # HGST HUH721212ALN604 - Bi-LSTM weights
│   ├── gru_HGST_HUH721212ALN604_best.pth     # HGST HUH721212ALN604 - Bi-GRU weights
│   └── cnn1d_HGST_HUH721212ALN604_best.pth   # HGST HUH721212ALN604 - 1D-CNN weights
│
└── Offline Data & Tensor Compilation Scripts:
    ├── module_1_eda.py                       # Global imbalance & correlation audit
    ├── model_inventory_scanner.py            # Hardware prefix & failure density scanner
    ├── seagate_2d_imputation.py              # Chronological sorting & grouped ffill()
    ├── seagate_3d_tensor_compiler.py         # Per-drive Z-score scaling & 15-day windowing
    └── multivendor_tensor_pipeline.py        # Automated Toshiba & HGST 3D tensor compiler
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

If `requirements.txt` is not present, install the packages directly:

```bash
pip install streamlit torch numpy pandas shap matplotlib scikit-learn
```

### 3. Launch the Application

Run the Streamlit server from the directory containing `app.py` and the `.pth` weight files:

```bash
streamlit run app.py
```

The app opens automatically in your default browser at **http://localhost:8501**.

---

## User Guide: How to Operate the Dashboard

### Step 1: Configure Target Hardware and Architecture (Left Sidebar)

- **Target Hardware Vendor:** Choose one of the three supported enterprise drive models:
  - Seagate `ST4000DM000`
  - Toshiba `MG07ACA14TA`
  - HGST `HUH721212ALN604`
- **Neural Network Architecture:** Choose **BiLSTM**, **GRU**, or **CNN1D**.
- Check for the green status indicator in the sidebar (`Loaded: <weight_file>.pth`). It confirms that the matching PyTorch state dictionary is active in memory.

### Step 2: Run Live Diagnostics (Tab 1: Live Diagnostics)

You can evaluate drive health using either **uploaded CSV telemetry** or the **synthetic signal simulator**.

#### Option A: Upload Telemetry CSV (15-Day Sequence)

1. Select **Upload Telemetry CSV (15-Day Sequence)**.
2. **To test immediately:** click **Download Sample Healthy CSV** or **Download Sample Failing CSV**. This gives you a 15-day template matched to your selected vendor. Drag and drop it into the uploader.
3. **To upload your own or raw Backblaze CSV logs**, follow these rules:
   - The CSV must contain **at least 15 valid chronological rows**. If it has more, the engine automatically uses the **latest 15 days**.
   - If a date column is present, rows are automatically sorted chronologically before the window is extracted.
   - Both friendly headers and raw Backblaze column names are accepted:

| Hardware Vendor | Accepted Friendly Headers | Accepted Raw Backblaze Headers |
| --- | --- | --- |
| **Seagate ST4000DM000** | `SMART 5`, `SMART 9`, `SMART 187`, `SMART 188`, `SMART 197`, `SMART 198` | `smart_5_raw`, `smart_9_raw`, `smart_187_raw`, `smart_188_raw`, `smart_197_raw`, `smart_198_raw` |
| **Toshiba MG07ACA14TA** | `SMART 5`, `SMART 9`, `SMART 192`, `SMART 193`, `SMART 197`, `SMART 198` | `smart_5_raw`, `smart_9_raw`, `smart_192_raw`, `smart_193_raw`, `smart_197_raw`, `smart_198_raw` |
| **HGST HUH721212ALN604** | `SMART 5`, `SMART 9`, `SMART 192`, `SMART 193`, `SMART 197`, `SMART 198` | `smart_5_raw`, `smart_9_raw`, `smart_192_raw`, `smart_193_raw`, `smart_197_raw`, `smart_198_raw` |

4. **Z-Score Standardization toggle:**
   - **Leave unchecked** if you upload pre-standardized tensors, the sample CSV downloads, or slices exported from `X_tensor.npy`.
   - **Check the box** if you upload raw, unnormalized S.M.A.R.T. integer counts. `StandardScaler` will normalize the sequence before inference.
5. Click **Execute OS Diagnostics**.

#### Option B: Synthetic Signal Simulator

1. Select **Synthetic Signal Simulator**.
2. Choose one of:
   - **Simulate Healthy Drive:** normal operational variance, `N(0, 0.5²)`.
   - **Simulate Failing Drive:** injects a **+3.5σ** degradation spike across **days 11–15**.
3. Click **Execute OS Diagnostics**.

#### Interpreting Diagnostic Outputs

| Output | Meaning |
| --- | --- |
| **Health Status Banner** | Red **CRITICAL ALARM: HARDWARE FAILURE PREDICTED** if the Sigmoid probability p̂ > 0.50. Green **DRIVE HEALTH OPTIMAL** if p̂ ≤ 0.50. |
| **Calculated Degradation Probability** | The exact Sigmoid probability, σ(z) × 100. |
| **OS Daemon Inference Latency** | Forward-pass CPU execution time in milliseconds. |
| **XAI Sensor Analysis** | Horizontal bar chart of cumulative SHAP values across the 6 S.M.A.R.T. attributes, showing which physical register drove the decision. |

### Step 3: Explore Dataset Demographics (Tab 2: Dataset Analytics)

- View the total compiled 15-day sequences, healthy sequences, and verified failure counts for the selected hardware model (Backblaze Q1 2023).
- Use the physical degradation table, which maps each S.M.A.R.T. attribute ID to its underlying electromechanical failure mechanism.

### Step 4: Audit Model Evaluation Metrics (Tab 3: Model Performance)

- **Row 1 (Standard & Global Classification Metrics):** Overall Accuracy, Balanced Accuracy, Specificity (True Negative Rate), and Weighted F1-Score.
- **Row 2 (Minority-Class & Cost-Sensitive Metrics):** Failure Recall (Sensitivity), False Alarm Rate (FAR) against the **< 0.1%** data-center target, Failure Precision / F1-Score, and the exact Dynamic Penalty Weight (`pos_weight`).
- **Detailed Diagnostic Report:** The 2×2 Confusion Matrix (TN, FP, FN, TP) alongside the class-wise Precision, Recall, F1-Score, and Support table.
- **Master 9-Model Cross-Vendor Benchmark:** A unified 11-column comparison table across all 9 vendor-architecture combinations.

---

## Master 9-Model Benchmark Summary

Best result for each vendor is shown in **bold**.

| Hardware Vendor (Dataset) | Architecture | Overall Acc (%) | Balanced Acc (%) | Recall (%) | Specificity (%) | FAR (%) | Precision (Fail) | F1-Score (Fail) | Macro F1 | Weighted F1 | Confusion Matrix (TN, FP, FN, TP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Seagate ST4000DM000** | **Bi-LSTM** | **99.9674** | **99.98** | **100.00** | **99.9674** | **0.0326** | **0.1964** | **0.3284** | **0.6641** | **0.9998** | 275632, 90, 0, 22 |
| Seagate ST4000DM000 | Bi-GRU | 99.9075 | 99.95 | 100.00 | 99.9075 | 0.0925 | 0.0794 | 0.1472 | 0.5733 | 0.9995 | 275467, 255, 0, 22 |
| Seagate ST4000DM000 | 1D-CNN | 99.9362 | 93.15 | 86.36 | 99.9373 | 0.0627 | 0.0990 | 0.1776 | 0.5886 | 0.9996 | 275549, 173, 3, 19 |
| **Toshiba MG07ACA14TA** | **Bi-LSTM** | 99.9007 | **89.31** | **78.72** | 99.9025 | 0.0975 | 0.0617 | 0.1144 | 0.5569 | 0.9994 | 576637, 563, 10, 37 |
| **Toshiba MG07ACA14TA** | **Bi-GRU** | **99.9298** | 87.20 | 74.47 | **99.9319** | **0.0681** | **0.0818** | **0.1474** | **0.5735** | **0.9996** | 576807, 393, 12, 35 |
| Toshiba MG07ACA14TA | 1D-CNN | 99.8992 | 71.23 | 42.55 | 99.9038 | 0.0962 | 0.0348 | 0.0643 | 0.5319 | 0.9994 | 576645, 555, 27, 20 |
| HGST HUH721212ALN604 | Bi-LSTM | 99.9431 | 99.97 | 100.00 | 99.9431 | 0.0569 | 0.0417 | 0.0800 | 0.5399 | 0.9997 | 161573, 92, 0, 4 |
| **HGST HUH721212ALN604** | **Bi-GRU** | **99.9629** | **99.98** | **100.00** | **99.9629** | **0.0371** | **0.0625** | **0.1176** | **0.5587** | **0.9998** | 161605, 60, 0, 4 |
| HGST HUH721212ALN604 | 1D-CNN | 99.8856 | 99.94 | 100.00 | 99.8856 | 0.1144 | 0.0212 | 0.0415 | 0.5204 | 0.9994 | 161480, 185, 0, 4 |

> **Note:** Overall accuracy is high for every model because of the extreme class imbalance. Use **Balanced Accuracy, Failure Recall, FAR, and Failure F1** to judge real failure-detection quality.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Sidebar shows the weight file is missing | Make sure all 9 `.pth` files are in the same folder as `app.py`, with the exact file names listed above. |
| CSV upload is rejected | The CSV needs at least 15 valid rows and the 6 required columns for the selected vendor (friendly or raw Backblaze headers). |
| Predictions look wrong on a raw CSV | Tick the **Z-Score Standardization** checkbox for raw, unnormalized values. Leave it unticked for pre-standardized data and the sample CSVs. |
| Port 8501 is busy | Run `streamlit run app.py --server.port 8502`. |

---

## Deploying to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and the 9 `.pth` weight files to a GitHub repository. **Do not push** the multi-gigabyte raw CSVs or `.npy` tensor archives.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and click **Create app**.
3. Select your GitHub repository and branch (`main`), and set `app.py` as the main file path.
4. Click **Deploy**.

---

## Authors

- **Tejas Khare** — School of Computer Science and Engineering, Vellore Institute of Technology ([tejas.khare2024@vitstudent.ac.in](mailto:tejas.khare2024@vitstudent.ac.in))
- **Yash Lohiya** — School of Computer Science and Engineering, Vellore Institute of Technology ([yash.lohiya2024@vitstudent.ac.in](mailto:yash.lohiya2024@vitstudent.ac.in))
