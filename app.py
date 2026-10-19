import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- 1. Deep Learning Architectures ---
class HDDBiLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(HDDBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class HDDGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(HDDGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class HDDCNN1D(nn.Module):
    def __init__(self, input_size=6, num_filters=64):
        super(HDDCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters*2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# --- 2. Configuration & State ---
st.set_page_config(page_title="Universal OS Diagnostic Engine", layout="wide")
st.title("Universal S.M.A.R.T. Degradation Probability Engine")
st.markdown("---")

HW_CONFIG = {
    'Seagate ST4000DM000': {
        'id': 'ST4000DM000',
        'features': ['SMART 5', 'SMART 9', 'SMART 187', 'SMART 188', 'SMART 197', 'SMART 198'],
        'raw_cols': ['smart_5_raw', 'smart_9_raw', 'smart_187_raw', 'smart_188_raw', 'smart_197_raw', 'smart_198_raw'],
        'total': "1,378,717", 'healthy': "1,378,573", 'failed': "144"
    },
    'Toshiba MG07ACA14TA': {
        'id': 'TOSHIBA_MG07ACA14TA',
        'features': ['SMART 5', 'SMART 9', 'SMART 192', 'SMART 193', 'SMART 197', 'SMART 198'],
        'raw_cols': ['smart_5_raw', 'smart_9_raw', 'smart_192_raw', 'smart_193_raw', 'smart_197_raw', 'smart_198_raw'],
        'total': "2,886,234", 'healthy': "2,886,147", 'failed': "87"
    },
    'HGST HUH721212ALN604': {
        'id': 'HGST_HUH721212ALN604',
        'features': ['SMART 5', 'SMART 9', 'SMART 192', 'SMART 193', 'SMART 197', 'SMART 198'],
        'raw_cols': ['smart_5_raw', 'smart_9_raw', 'smart_192_raw', 'smart_193_raw', 'smart_197_raw', 'smart_198_raw'],
        'total': "808,343", 'healthy': "808,274", 'failed': "69"
    }
}

PERFORMANCE_DATA = {
    'Seagate ST4000DM000': {
        'BiLSTM': {'far': '0.0326%', 'recall': '100.00%', 'cm': [[275632, 90], [0, 22]], 'cr': {'precision': ['1.0000', '0.1964', '-', '0.5982', '0.9999'], 'recall': ['0.9997', '1.0000', '-', '0.9998', '0.9997'], 'f1': ['0.9998', '0.3284', '0.9997', '0.6641', '0.9998'], 'support': ['275,722', '22', '275,744', '275,744', '275,744']}},
        'GRU': {'far': '0.0925%', 'recall': '100.00%', 'cm': [[275467, 255], [0, 22]], 'cr': {'precision': ['1.0000', '0.0794', '-', '0.5397', '0.9999'], 'recall': ['0.9991', '1.0000', '-', '0.9995', '0.9991'], 'f1': ['0.9995', '0.1472', '0.9991', '0.5733', '0.9995'], 'support': ['275,722', '22', '275,744', '275,744', '275,744']}},
        'CNN1D': {'far': '0.0627%', 'recall': '86.36%', 'cm': [[275549, 173], [3, 19]], 'cr': {'precision': ['1.0000', '0.0990', '-', '0.5495', '0.9999'], 'recall': ['0.9994', '0.8636', '-', '0.9315', '0.9994'], 'f1': ['0.9997', '0.1776', '0.9994', '0.5886', '0.9996'], 'support': ['275,722', '22', '275,744', '275,744', '275,744']}}
    },
    'Toshiba MG07ACA14TA': {
        'BiLSTM': {'far': '0.0975%', 'recall': '78.72%', 'cm': [[576637, 563], [10, 37]], 'cr': {'precision': ['1.0000', '0.0617', '-', '0.5308', '0.9999'], 'recall': ['0.9990', '0.7872', '-', '0.8931', '0.9990'], 'f1': ['0.9995', '0.1144', '0.9990', '0.5569', '0.9994'], 'support': ['577,200', '47', '577,247', '577,247', '577,247']}},
        'GRU': {'far': '0.0681%', 'recall': '74.47%', 'cm': [[576807, 393], [12, 35]], 'cr': {'precision': ['1.0000', '0.0818', '-', '0.5409', '0.9999'], 'recall': ['0.9993', '0.7447', '-', '0.8720', '0.9993'], 'f1': ['0.9996', '0.1474', '0.9993', '0.5735', '0.9996'], 'support': ['577,200', '47', '577,247', '577,247', '577,247']}},
        'CNN1D': {'far': '0.0962%', 'recall': '42.55%', 'cm': [[576645, 555], [27, 20]], 'cr': {'precision': ['1.0000', '0.0348', '-', '0.5174', '0.9999'], 'recall': ['0.9990', '0.4255', '-', '0.7123', '0.9990'], 'f1': ['0.9995', '0.0643', '0.9990', '0.5319', '0.9994'], 'support': ['577,200', '47', '577,247', '577,247', '577,247']}}
    },
    'HGST HUH721212ALN604': {
        'BiLSTM': {'far': '0.0569%', 'recall': '100.00%', 'cm': [[161573, 92], [0, 4]], 'cr': {'precision': ['1.0000', '0.0417', '-', '0.5208', '1.0000'], 'recall': ['0.9994', '1.0000', '-', '0.9997', '0.9994'], 'f1': ['0.9997', '0.0800', '0.9994', '0.5399', '0.9997'], 'support': ['161,665', '4', '161,669', '161,669', '161,669']}},
        'GRU': {'far': '0.0371%', 'recall': '100.00%', 'cm': [[161605, 60], [0, 4]], 'cr': {'precision': ['1.0000', '0.0625', '-', '0.5312', '1.0000'], 'recall': ['0.9996', '1.0000', '-', '0.9998', '0.9996'], 'f1': ['0.9998', '0.1176', '0.9996', '0.5587', '0.9998'], 'support': ['161,665', '4', '161,669', '161,669', '161,669']}},
        'CNN1D': {'far': '0.1144%', 'recall': '100.00%', 'cm': [[161480, 185], [0, 4]], 'cr': {'precision': ['1.0000', '0.0212', '-', '0.5106', '1.0000'], 'recall': ['0.9989', '1.0000', '-', '0.9994', '0.9989'], 'f1': ['0.9994', '0.0415', '0.9989', '0.5204', '0.9994'], 'support': ['161,665', '4', '161,669', '161,669', '161,669']}}
    }
}

# --- Global Sidebar Navigation ---
with st.sidebar:
    st.header("Diagnostic Configuration")
    target_hw = st.selectbox("Target Hardware Vendor", list(HW_CONFIG.keys()))
    target_arch = st.selectbox("Neural Network Architecture", ["BiLSTM", "GRU", "CNN1D"])
    
    hw_id = HW_CONFIG[target_hw]['id']
    feature_names = HW_CONFIG[target_hw]['features']
    raw_col_names = HW_CONFIG[target_hw]['raw_cols']
    weight_file = f"{target_arch.lower()}_{hw_id}_best.pth"
    if target_hw == 'Seagate ST4000DM000' and target_arch == 'BiLSTM':
        weight_file = 'bilstm_hdd_best.pth'

@st.cache_resource
def load_model(arch, weights_path):
    if arch == "BiLSTM": model = HDDBiLSTM()
    elif arch == "GRU": model = HDDGRU()
    elif arch == "CNN1D": model = HDDCNN1D()
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
        st.sidebar.success(f"Loaded: {weights_path}")
    except FileNotFoundError:
        st.sidebar.error(f"Missing weights: {weights_path}")
        
    model.eval()
    return model

def generate_sample_csv(cols, is_failing=False):
    np.random.seed(42 if not is_failing else 99)
    data = np.random.randn(15, 6) * 0.4
    if is_failing:
        data[10:, :] += 3.5
    df = pd.DataFrame(np.round(data, 4), columns=cols)
    df.insert(0, 'day', [f"Day {i+1}" for i in range(15)])
    return df.to_csv(index=False).encode('utf-8')

model = load_model(target_arch, weight_file)

# --- UI Layout ---
tab1, tab2, tab3 = st.tabs(["Live Diagnostics", "Dataset Analytics", "Model Performance"])

# TAB 1: Diagnostics
with tab1:
    st.subheader(f"Target Telemetry: {target_hw} via {target_arch}")
    
    input_mode = st.radio(
        "Select Telemetry Input Source:",
        ("Upload Telemetry CSV (15-Day Sequence)", "Synthetic Signal Simulator"),
        horizontal=True
    )
    
    sample_tensor = None
    
    if input_mode == "Upload Telemetry CSV (15-Day Sequence)":
        col_dl1, col_dl2, _ = st.columns([1, 1, 2])
        with col_dl1:
            st.download_button(
                label="Download Sample Healthy CSV",
                data=generate_sample_csv(feature_names, is_failing=False),
                file_name=f"sample_healthy_{hw_id}.csv",
                mime="text/csv"
            )
        with col_dl2:
            st.download_button(
                label="Download Sample Failing CSV",
                data=generate_sample_csv(feature_names, is_failing=True),
                file_name=f"sample_failing_{hw_id}.csv",
                mime="text/csv"
            )
            
        uploaded_file = st.file_uploader(
            f"Upload a CSV file containing at least 15 days of S.M.A.R.T. logs ({', '.join(feature_names)})",
            type=["csv"]
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                if 'date' in df_upload.columns:
                    df_upload = df_upload.sort_values('date')
                
                if all(col in df_upload.columns for col in feature_names):
                    extracted_df = df_upload[feature_names].copy()
                elif all(col in df_upload.columns for col in raw_col_names):
                    extracted_df = df_upload[raw_col_names].copy()
                    extracted_df.columns = feature_names
                else:
                    st.error(f"CSV must contain either {feature_names} or {raw_col_names}.")
                    extracted_df = None
                
                if extracted_df is not None:
                    extracted_df = extracted_df.ffill().dropna()
                    if len(extracted_df) < 15:
                        st.error(f"Uploaded file has only {len(extracted_df)} valid rows. At least 15 chronological days are required.")
                    else:
                        scale_raw = st.checkbox(
                            "Apply Z-Score Standardization (Enable if uploading unscaled raw S.M.A.R.T. counts)",
                            value=False
                        )
                        if scale_raw:
                            scaler = StandardScaler()
                            scaled_vals = scaler.fit_transform(extracted_df.values)
                            window_vals = scaled_vals[-15:]
                        else:
                            window_vals = extracted_df.tail(15).values
                            
                        with st.expander("Preview Extracted 15-Day Telemetry Window", expanded=True):
                            preview_df = pd.DataFrame(window_vals, columns=feature_names)
                            preview_df.index = [f"Day {i+1}" for i in range(15)]
                            st.dataframe(preview_df, use_container_width=True)
                            
                        sample_tensor = torch.tensor(window_vals, dtype=torch.float32).unsqueeze(0)
            except Exception as e:
                st.error(f"Error parsing CSV file: {e}")
    else:
        test_type = st.radio(
            "Select 15-Day Synthetic Profile:",
            ("Simulate Healthy Drive", "Simulate Failing Drive"),
            horizontal=True
        )
        if test_type == "Simulate Healthy Drive":
            sample_tensor = torch.randn(1, 15, 6) * 0.5 
        else:
            sample_tensor = torch.randn(1, 15, 6) 
            sample_tensor[:, 10:, :] += 3.5 

    if st.button("Execute OS Diagnostics", type="primary", disabled=(sample_tensor is None)):
        with st.spinner("Analyzing temporal degradation sequences..."):
            t_start = time.perf_counter()
            with torch.no_grad():
                output = model(sample_tensor)
                probability = torch.sigmoid(output).item()
            inference_ms = (time.perf_counter() - t_start) * 1000.0
                
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if probability > 0.5:
                    st.error("CRITICAL ALARM: HARDWARE FAILURE PREDICTED")
                else:
                    st.success("DRIVE HEALTH OPTIMAL")
                
                st.write(f"**Calculated Degradation Probability:** {probability * 100:.2f}%")
                st.write(f"**OS Daemon Inference Latency:** {inference_ms:.2f} ms")
            
            with col2:
                st.write("**Explainable AI (XAI) Sensor Analysis**")
                try:
                    background_data = torch.randn(100, 15, 6)
                    explainer = shap.GradientExplainer(model, background_data)
                    
                    tensor_for_shap = sample_tensor.clone().detach().requires_grad_(True)
                    shap_values = explainer.shap_values(tensor_for_shap)
                    
                    if isinstance(shap_values, list): shap_values = shap_values[0]
                    if torch.is_tensor(shap_values): shap_values = shap_values.detach().cpu().numpy()
                    else: shap_values = np.array(shap_values)
                        
                    shap_values = shap_values.reshape(-1, 6)
                    feature_importance = [float(val) for val in np.abs(shap_values).sum(axis=0).flatten()]
                    
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.barh(feature_names, feature_importance, color='#ff4b4b' if probability > 0.5 else '#00cc66')
                    ax.set_xlabel("Cumulative SHAP Value")
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception:
                    st.warning("SHAP execution unavailable for this architecture tensor mapping.")

# TAB 2: Analytics
with tab2:
    st.subheader(f"Backblaze Q1 2023 Demographics ({target_hw})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sequences Analyzed", HW_CONFIG[target_hw]['total'])
    col2.metric("Healthy Sequences", HW_CONFIG[target_hw]['healthy'])
    col3.metric("Verified Hardware Failures", HW_CONFIG[target_hw]['failed'])
    st.info("Note: The extreme class imbalance necessitates dynamic cost-sensitive learning over synthetic oversampling.")
    
    st.markdown("---")
    st.subheader("Vendor-Specific Feature Engineering")
    st.write(f"Displaying isolated mechanical degradation indicators mapped for **{target_hw}**:")
    
    desc = []
    for feat in feature_names:
        if feat == 'SMART 5': desc.append("Physical damage to disk platters; sectors remapped.")
        elif feat == 'SMART 9': desc.append("Total operational age of the drive.")
        elif feat == 'SMART 187': desc.append("Hardware read errors the ECC cannot fix.")
        elif feat == 'SMART 188': desc.append("Aborted operations due to unresponsive hardware.")
        elif feat == 'SMART 192': desc.append("Power-off retract count (Emergency head parking).")
        elif feat == 'SMART 193': desc.append("Load/Unload cycle count of the actuator arm.")
        elif feat == 'SMART 197': desc.append("Sectors waiting to be remapped because they are failing.")
        elif feat == 'SMART 198': desc.append("Sectors that could not be read or written to.")
        
    feature_df = pd.DataFrame({
        "Attribute": feature_names,
        "Physical Degradation Representation": desc
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

# TAB 3: Performance
with tab3:
    st.subheader(f"Evaluation Metrics: {target_hw} via {target_arch}")
    
    metrics = PERFORMANCE_DATA[target_hw][target_arch]
    tn, fp = metrics['cm'][0]
    fn, tp = metrics['cm'][1]
    total_samples = tn + fp + fn + tp
    
    # Exact Mathematical Derivations from Confusion Matrix
    overall_acc = ((tn + tp) / total_samples) * 100.0
    specificity = (tn / (tn + fp)) * 100.0
    sensitivity = (tp / (tp + fn)) * 100.0
    balanced_acc = (sensitivity + specificity) / 2.0
    failure_precision = float(metrics['cr']['precision'][1])
    failure_f1 = float(metrics['cr']['f1'][1])
    macro_f1 = float(metrics['cr']['f1'][3])
    weighted_f1 = float(metrics['cr']['f1'][4])
    pos_weight = float(HW_CONFIG[target_hw]['healthy'].replace(',','')) / float(HW_CONFIG[target_hw]['failed'].replace(',',''))
    
    # Row 1: Standard Classification Metrics (What the Professor Expects)
    st.markdown("##### Standard & Global Classification Metrics")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Overall Accuracy", f"{overall_acc:.4f}%")
    r1c2.metric("Balanced Accuracy", f"{balanced_acc:.2f}%")
    r1c3.metric("Specificity (TNR)", f"{specificity:.4f}%")
    r1c4.metric("Weighted F1-Score", f"{weighted_f1:.4f}")
    
    # Row 2: Anomaly & Minority-Class Metrics (Domain-Specific Metrics)
    st.markdown("##### Minority-Class (Failure) & Cost-Sensitive Metrics")
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Failure Recall (Sensitivity)", metrics['recall'])
    r2c2.metric("False Alarm Rate (FAR)", metrics['far'], delta="< 0.1% Target", delta_color="inverse")
    r2c3.metric("Failure Precision / F1", f"{failure_precision:.4f} / {failure_f1:.4f}")
    r2c4.metric("Dynamic Penalty (pos_weight)", f"{pos_weight:.2f}")
    
    st.markdown("---")
    st.subheader("Detailed Diagnostic Report")
    
    col_cm, col_cr = st.columns(2)
    with col_cm:
        st.write("**Confusion Matrix**")
        cm_df = pd.DataFrame(
            metrics['cm'],
            index=["Actual Healthy (0)", "Actual Failure (1)"],
            columns=["Predicted Healthy (0)", "Predicted Failure (1)"]
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(f"True Negatives: {tn:,} | False Positives: {fp:,} | False Negatives: {fn:,} | True Positives: {tp:,}")
        
    with col_cr:
        st.write("**Classification Report**")
        cr_df = pd.DataFrame(
            {
                "Precision": metrics['cr']['precision'],
                "Recall": metrics['cr']['recall'],
                "F1-Score": metrics['cr']['f1'],
                "Support": metrics['cr']['support']
            },
            index=["Class 0.0 (Healthy)", "Class 1.0 (Failure)", "Accuracy", "Macro Avg", "Weighted Avg"]
        )
        st.dataframe(cr_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Master 9-Model Cross-Vendor Benchmark")
    benchmark_rows = []
    for hw_name, arch_dict in PERFORMANCE_DATA.items():
        for arch_name, m_data in arch_dict.items():
            m_tn, m_fp = m_data['cm'][0]
            m_fn, m_tp = m_data['cm'][1]
            m_tot = m_tn + m_fp + m_fn + m_tp
            m_acc = ((m_tn + m_tp) / m_tot) * 100.0
            m_spec = (m_tn / (m_tn + m_fp)) * 100.0
            m_rec = (m_tp / (m_tp + m_fn)) * 100.0
            m_bal = (m_rec + m_spec) / 2.0
            benchmark_rows.append({
                "Hardware Vendor": hw_name,
                "Architecture": arch_name,
                "Accuracy (%)": f"{m_acc:.4f}%",
                "Balanced Acc (%)": f"{m_bal:.2f}%",
                "Recall / Sensitivity (%)": m_data['recall'],
                "Specificity (%)": f"{m_spec:.4f}%",
                "FAR (%)": m_data['far'],
                "Precision (Failure)": m_data['cr']['precision'][1],
                "F1-Score (Failure)": m_data['cr']['f1'][1],
                "Macro F1": m_data['cr']['f1'][3],
                "Weighted F1": m_data['cr']['f1'][4]
            })
    st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)