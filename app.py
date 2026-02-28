import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import json
import io

# App Title & Context from Case Study
st.title("NexusBank: CreditGuard AI Validation Portal")
st.markdown("### Lab 6: Robustness & Validation Stress-Testing")

# 1. UPLOAD SECTION
st.sidebar.header("Upload Validation Assets")
uploaded_data = st.sidebar.file_uploader("Upload Baseline Data (CSV)", type="csv")
uploaded_model = st.sidebar.file_uploader("Upload XGBoost Model (.json or .bin)", type=["json", "bin"])

if uploaded_data and uploaded_model:
    # Load Data
    df = pd.read_csv(uploaded_data)
    
    # Load Model (using XGBoost's native loading)
    model = xgb.XGBClassifier()
    model.load_model(uploaded_model)
    
    # Identify Features (Case Study: Income, DTI, Credit History, Stability)
    # For this example, we assume the last column is the 'target'
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    st.success("Assets loaded successfully. Ready to establish baseline.")

    # --- STEP 1: ESTABLISH THE BASELINE ---
    baseline_probs = model.predict_proba(X)[:, 1]
    baseline_auc = roc_auc_score(y, baseline_probs)
    baseline_brier = brier_score_loss(y, baseline_probs)
    
    st.metric("Golden Baseline AUC", f"{baseline_auc:.3f}")
    st.metric("Golden Baseline Brier Score", f"{baseline_brier:.3f}")

    # --- STEP 2: STRESS SCENARIOS ---
    st.header("Stress Test Analysis")
    
    # Scenario 1: Gaussian Noise Injection [cite: 56]
    X_noise = X.copy()
    noise_factor = st.slider("Select Noise Factor (σ)", 0.0, 0.5, 0.1)
    for col in X.select_dtypes(include=[np.number]).columns:
        noise = np.random.normal(0, X[col].std() * noise_factor, size=len(X))
        X_noise[col] += noise
    
    # Scenario 2: Economic Shock (Feature Scaling) [cite: 67]
    X_shock = X.copy()
    # Case study uses a shift factor of 0.5 for financial features [cite: 71]
    financial_features = [col for col in X.columns if 'income' in col.lower() or 'asset' in col.lower()]
    X_shock[financial_features] = X_shock[financial_features] * 0.5

    # --- STEP 3: QUANTIFY DEGRADATION ---
    shock_probs = model.predict_proba(X_shock)[:, 1]
    shock_auc = roc_auc_score(y, shock_probs)
    shock_brier = brier_score_loss(y, shock_probs)
    
    auc_drop = (baseline_auc - shock_auc) / baseline_auc
    
    # Display Results
    st.subheader("Economic Shock Results")
    st.write(f"Stressed AUC: {shock_auc:.3f}")
    
    if shock_auc < 0.70:
        st.error(f"CRITICAL FAILURE: AUC dropped below 0.70 threshold [cite: 88]")
    else:
        st.success("Economic Shock Resilience: PASS")

    # --- STEP 4: GENERATE EVIDENCE ---
    st.header("Generate Validation Evidence")
    
    # JSON Violation List 
    violations = [
        {
            "id": "CRIT_01",
            "type": "Economic Shock Resilience",
            "status": "FAIL" if shock_auc < 0.70 else "PASS",
            "metric": "AUC",
            "value": float(shock_auc)
        }
    ]
    
    # Markdown Summary 
    summary_md = f"""# Executive Summary: CreditGuard AI
## Verdict: {"NO GO" if shock_auc < 0.70 else "GO"}
- **Baseline AUC:** {baseline_auc:.3f}
- **Shock AUC:** {shock_auc:.3f}
- **Degradation:** {auc_drop*100:.1f}%
"""

    # Download Buttons
    st.download_button("Download violations_list.json", json.dumps(violations), "violations_list.json")
    st.download_button("Download executive_summary.md", summary_md, "executive_summary.md")

else:
    st.info("Please upload both the baseline data and the model file in the sidebar to begin.")