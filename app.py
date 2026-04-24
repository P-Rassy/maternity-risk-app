# =============================================================================
# app.py — Maternity Complication Risk Predictor
# =============================================================================

import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# CUSTOM FUNCTIONS REQUIRED BY THE PREPROCESSOR
# =============================================================================

def to_str_array(x):
    """Required by the saved preprocessor pipeline."""
    x = pd.DataFrame(x).astype(object)
    x = x.where(~x.isna(), "__MISSING__")
    return x.astype(str).values

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Maternity Risk Predictor",
    page_icon="🤱",
    layout="wide",
)

# =============================================================================
# LOAD MODEL ARTEFACTS
# =============================================================================

@st.cache_resource
def load_artefacts():
    model        = joblib.load("calibrated_xgb.joblib")
    preprocessor = joblib.load("xgb_preprocessor.joblib")
    with open("model_artefacts.json") as f:
        meta = json.load(f)
    return model, preprocessor, meta

try:
    model, preprocessor, meta = load_artefacts()
    THRESHOLD      = meta["stack_threshold"]
    FINAL_FEATURES = meta["final_features"]
    MODEL_LOADED   = True
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")
    st.info(
        "Make sure calibrated_xgb.joblib, xgb_preprocessor.joblib, "
        "and model_artefacts.json are in the same folder as app.py"
    )
    st.stop()

# =============================================================================
# HELPERS
# =============================================================================

def predict(input_df):
    X_trans = preprocessor.transform(input_df)
    prob    = model.predict_proba(X_trans)[:, 1][0]
    flag    = prob >= THRESHOLD
    return prob, flag


def get_shap_values(input_df):
    X_trans   = preprocessor.transform(input_df)
    xgb_est   = model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(xgb_est)
    sv        = explainer.shap_values(X_trans)
    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = [f"feature_{i}" for i in range(X_trans.shape[1])]
    return sv[0], feat_names, X_trans[0]

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📊 Model Info")
    st.markdown("---")
    st.metric("ROC-AUC",   f"{meta.get('stack_auc_test',  0):.3f}")
    st.metric("Recall",    f"{meta.get('stack_recall',    0):.3f}")
    st.metric("Precision", f"{meta.get('stack_precision', 0):.3f}")
    st.markdown("---")
    st.caption(f"Decision threshold : **{THRESHOLD:.3f}**")
    st.caption(f"Features used      : **{len(FINAL_FEATURES)}**")
    st.caption("Model : XGBoost + isotonic calibration")
    st.markdown("---")
    st.warning(
        "⚠️ This tool is for **clinical decision support only**. "
        "It does not replace professional medical judgment."
    )

# =============================================================================
# MAIN PAGE
# =============================================================================

st.title("🤱 Maternity Complication Risk Predictor")
st.markdown(
    "Enter patient information below and click **Run Prediction** "
    "to assess the risk of maternity complications."
)
st.markdown("---")

# =============================================================================
# INPUT FORM
# =============================================================================

with st.form("patient_form"):

    # ── Demographics ──────────────────────────────────────────────────────────
    st.subheader("👤 Patient Demographics")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Age (years)",
            min_value=15, max_value=50, value=28, step=1
        )
    with c2:
        bmi_first_visit = st.number_input(
            "BMI at first visit",
            min_value=15.0, max_value=60.0,
            value=24.0, step=0.1, format="%.1f"
        )
    with c3:
        weight_gain = st.number_input(
            "Weight gain so far (kg)",
            min_value=0.0, max_value=50.0,
            value=10.0, step=0.5, format="%.1f"
        )

    st.markdown("---")

    # ── Obstetric history ─────────────────────────────────────────────────────
    st.subheader("🤰 Obstetric History")
    c4, c5, c6 = st.columns(3)

    with c4:
        prev_pregnancies = st.number_input(
            "Previous pregnancies",
            min_value=0, max_value=12, value=0, step=1
        )
    with c5:
        prev_deliveries = st.number_input(
            "Previous deliveries",
            min_value=0, max_value=10, value=0, step=1
        )
    with c6:
        prev_c_sections = st.number_input(
            "Previous C-sections",
            min_value=0, max_value=6, value=0, step=1
        )

    st.markdown("---")

    # ── Current pregnancy ─────────────────────────────────────────────────────
    st.subheader("🏥 Current Pregnancy")
    c7, c8 = st.columns(2)

    with c7:
        multiple_gestation = st.selectbox(
            "Multiple gestation?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        num_fetuses = st.number_input(
            "Number of fetuses",
            min_value=1, max_value=6, value=1, step=1
        )

    with c8:
        third_trimester_anomalies = st.selectbox(
            "Third trimester anomalies?",
            options=[
                "none", "growth anomaly",
                "pelvic deformity", "other"
            ]
        )
        prev_comorbidities = st.selectbox(
            "Previous comorbidities?",
            options=[
                "none", "diabetes", "hypertension",
                "cardiac pulmonary", "permanent cerclage", "other"
            ]
        )

    st.markdown("---")

    # ── PROM scores ───────────────────────────────────────────────────────────
    st.subheader("📋 PROM Score Changes (3rd − 1st trimester)")
    c9, c10, c11 = st.columns(3)

    with c9:
        wexner_change = st.number_input(
            "Wexner score change",
            min_value=-20.0, max_value=20.0,
            value=0.0, step=0.5,
            help="Positive = bowel function worsened"
        )
    with c10:
        phq2_change = st.number_input(
            "PHQ-2 score change",
            min_value=-6.0, max_value=6.0,
            value=0.0, step=0.5,
            help="Positive = depression symptoms worsened"
        )
    with c11:
        poor_health_1st = st.selectbox(
            "Poor self-rated health at first visit?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submitted = st.form_submit_button(
        "🔍 Run Prediction", use_container_width=True
    )

# =============================================================================
# PREDICTION + RESULTS
# =============================================================================

if submitted:

    # ── Derived features ──────────────────────────────────────────────────────
    bmi_x_weight_gain = bmi_first_visit * weight_gain
    age_x_bmi         = age * bmi_first_visit

    # ── Build input row ───────────────────────────────────────────────────────
    input_dict = {
        "age":                                        age,
        "age_x_bmi":                                  age_x_bmi,
        "bmi_first_visit":                            bmi_first_visit,
        "bmi_x_weight_gain":                          bmi_x_weight_gain,
        "multiple_gestation__third_trimester":        multiple_gestation,
        "num_fetuses__third_trimester":               num_fetuses,
        "poor_health_1st":                            poor_health_1st,
        "prev_c_sections__first_visit":               prev_c_sections,
        "prev_comorbidities__first_visit":            prev_comorbidities,
        "prev_deliveries__first_visit":               prev_deliveries,
        "prev_pregnancies__first_visit":              prev_pregnancies,
        "third_trimester_anomalies__third_trimester": third_trimester_anomalies,
        "wexner_change":                              wexner_change,
    }

    # Fill any extra features the model expects with 0
    for f in FINAL_FEATURES:
        if f not in input_dict:
            input_dict[f] = 0

    input_df = pd.DataFrame([input_dict])[FINAL_FEATURES]

    # ── Run model ─────────────────────────────────────────────────────────────
    prob, flag = predict(input_df)

    # ── Result header ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            label       = "Complication Risk Score",
            value       = f"{prob:.1%}",
            delta       = f"Threshold: {THRESHOLD:.1%}",
            delta_color = "off",
        )
    with r2:
        if flag:
            st.error("🔴 HIGH RISK — Patient flagged for review")
        else:
            st.success("🟢 LOW RISK — No immediate flag")
    with r3:
        distance   = abs(prob - THRESHOLD)
        confidence = (
            "High"     if distance > 0.15 else
            "Moderate" if distance > 0.07 else
            "Low"
        )
        st.metric("Prediction confidence", confidence)

    # ── Risk gauge bar ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 1.2))
    bar_color = "#C0392B" if flag else "#27AE60"
    ax.barh(["Risk"], [prob],     color=bar_color, height=0.5)
    ax.barh(["Risk"], [1 - prob], left=[prob], color="#EEEEEE", height=0.5)
    ax.axvline(
        THRESHOLD, color="#2C3E50", lw=2, ls="--",
        label=f"Threshold ({THRESHOLD:.2f})"
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    st.pyplot(fig)
    plt.close()

    # ── SHAP explanation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 What is driving this prediction?")

    try:
        sv, feat_names, x_vals = get_shap_values(input_df)

        top_idx   = np.argsort(np.abs(sv))[::-1][:10]
        top_sv    = sv[top_idx]
        top_names = [feat_names[i] for i in top_idx]
        colors    = ["#C0392B" if v > 0 else "#2980B9" for v in top_sv]

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.barh(range(len(top_sv)), top_sv[::-1], color=colors[::-1])
        ax2.set_yticks(range(len(top_sv)))
        ax2.set_yticklabels(top_names[::-1], fontsize=9)
        ax2.axvline(0, color="black", lw=0.8)
        ax2.set_xlabel("SHAP value (impact on risk score)")
        ax2.set_title("Feature contributions for this patient")
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.legend(handles=[
            Patch(facecolor="#C0392B", label="Increases risk"),
            Patch(facecolor="#2980B9", label="Decreases risk"),
        ], fontsize=8)
        st.pyplot(fig2)
        plt.close()

        st.caption(
            "🔴 Red bars push the risk score **up**.  "
            "🔵 Blue bars push the risk score **down**."
        )

    except Exception as e:
        st.info(f"SHAP explanation unavailable for this patient: {e}")

    # ── Clinical notes summary ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Summary for Clinical Notes")

    st.code(f"""
Patient Risk Assessment — Maternity Complication Detection Model
----------------------------------------------------------------
Risk Level         : {"HIGH" if flag else "LOW"}
Risk Score         : {prob:.1%}
Decision Threshold : {THRESHOLD:.1%}
Model              : XGBoost + isotonic calibration
Model ROC-AUC      : {meta.get('stack_auc_test', '—')}

Key patient values entered:
  Age                  : {age} years
  BMI (first visit)    : {bmi_first_visit:.1f}
  Weight gain          : {weight_gain:.1f} kg
  BMI x Weight gain    : {bmi_x_weight_gain:.1f}
  Previous pregnancies : {prev_pregnancies}
  Previous deliveries  : {prev_deliveries}
  Previous C-sections  : {prev_c_sections}
  Wexner score change  : {wexner_change:+.1f}
  PHQ-2 score change   : {phq2_change:+.1f}
  Multiple gestation   : {"Yes" if multiple_gestation else "No"}
  Comorbidities        : {prev_comorbidities}
  3rd trim. anomalies  : {third_trimester_anomalies}

NOTE: This output is generated by a decision-support model.
Clinical judgment must be applied before any action is taken.
    """, language="text")
