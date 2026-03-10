import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import shap
import sqlite3
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import roc_curve, auc

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Breast Cancer Clinical System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- MEDICAL UI STYLE ---------------- #
st.markdown("""
<style>
.main {
    background-color: #f4f9ff;
}
h1, h2, h3 {
    color: #0a3d62;
}
.stButton>button {
    background-color: #0a3d62;
    color: white;
    height: 3em;
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("cancer_model.pkl")

model = load_model()

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("patients.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS records(
name TEXT,
age INTEGER,
gender TEXT,
ai_stage TEXT,
clinical_stage TEXT,
probability REAL,
date TEXT
)
""")
conn.commit()

# ---------------- AI STAGE CLASSIFICATION ---------------- #
def get_stage(probability):
    if probability < 0.25:
        return "Stage 1"
    elif probability < 0.50:
        return "Stage 2"
    elif probability < 0.75:
        return "Stage 3"
    else:
        return "Stage 4"

# ---------------- TNM CLINICAL STAGING ---------------- #
def calculate_stage(tumor_size, lymph_nodes, metastasis):
    if metastasis == 1:
        return "Stage IV – Distant Metastasis"
    elif tumor_size < 2 and lymph_nodes == 0:
        return "Stage I – Localized"
    elif tumor_size < 5:
        return "Stage II – Regional Spread"
    else:
        return "Stage III – Advanced Local"

# ---------------- STAGE DETAILS ---------------- #
def stage_details(stage):
    details = {
        "Stage 1": {
            "Symptoms": ["Small painless lump", "No lymph node spread", "Minor breast texture change"],
            "Diet": ["Leafy greens", "Berries", "High protein foods", "Whole grains"],
            "Precautions": ["Regular checkups", "Maintain healthy weight", "Avoid alcohol & smoking"]
        },
        "Stage 2": {
            "Symptoms": ["Growing lump", "Swollen lymph nodes", "Skin dimpling"],
            "Diet": ["Broccoli", "Carrots", "Omega-3 foods", "Vitamin C rich fruits"],
            "Precautions": ["Chemotherapy if advised", "Healthy lifestyle", "Light exercise"]
        },
        "Stage 3": {
            "Symptoms": ["Large tumor", "Redness & swelling", "Nipple inversion"],
            "Diet": ["Iron rich foods", "Soft protein meals", "Hydration"],
            "Precautions": ["Aggressive treatment", "Avoid infections", "Proper rest"]
        },
        "Stage 4": {
            "Symptoms": ["Bone pain", "Extreme fatigue", "Weight loss", "Breathing difficulty"],
            "Diet": ["High calorie diet", "Smoothies & soups", "Calcium rich foods"],
            "Precautions": ["Palliative care", "Pain management", "Emotional support"]
        }
    }
    return details.get(stage, details["Stage 1"])

# ---------------- FULL PDF GENERATOR ---------------- #
def create_pdf(name, age, gender, stage, clinical_stage, probability):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("MULTI-SPECIALITY HOSPITAL", styles["Heading1"]))
    elements.append(Paragraph("Breast Cancer Clinical Assessment Report", styles["Heading3"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Patient Information", styles["Heading2"]))
    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Cancer Risk Analysis", styles["Heading2"]))
    elements.append(Paragraph(f"AI Predicted Stage: {stage}", styles["Normal"]))
    elements.append(Paragraph(f"TNM Clinical Stage: {clinical_stage}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Probability: {probability:.2%}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    info = stage_details(stage)

    elements.append(Paragraph("Clinical Symptoms", styles["Heading2"]))
    elements.append(ListFlowable(
        [ListItem(Paragraph(i, styles["Normal"])) for i in info["Symptoms"]],
        bulletType='bullet'
    ))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Dietary Recommendations", styles["Heading2"]))
    elements.append(ListFlowable(
        [ListItem(Paragraph(i, styles["Normal"])) for i in info["Diet"]],
        bulletType='bullet'
    ))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Medical Precautions", styles["Heading2"]))
    elements.append(ListFlowable(
        [ListItem(Paragraph(i, styles["Normal"])) for i in info["Precautions"]],
        bulletType='bullet'
    ))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Doctor's Clinical Note:", styles["Heading2"]))
    elements.append(Paragraph(
        "This AI-based prediction is for preliminary screening only. "
        "Further diagnostic tests like biopsy, mammography, ultrasound, and imaging "
        "are required for medical confirmation. Clinical correlation is mandatory "
        "before final diagnosis and treatment planning.",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Authorized Signature: ____________________", styles["Normal"]))
    elements.append(Paragraph("Consultant Oncologist", styles["Normal"]))
    elements.append(Paragraph("Date: ____________________", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------- UI ---------------- #
st.markdown("## 🏥 Breast Cancer Disease Prediction System")
st.markdown("### Advanced Tumor Stage Prediction & Risk Assessment")
st.markdown("---")

st.sidebar.header("Patient Details")
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 1, 120, 30)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

st.sidebar.header("TNM Clinical Inputs")
tumor_size = st.sidebar.number_input("Tumor Size (cm)", 0.0, 10.0, 2.0)
lymph_nodes = st.sidebar.number_input("Lymph Nodes Affected", 0, 20, 0)
metastasis = st.sidebar.selectbox("Metastasis Present?", [0, 1])

st.sidebar.header("Tumor Measurements")

features = model.feature_names_in_
input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature.replace("_", " ").title(),
        value=0.0
    )

input_df = pd.DataFrame([input_data])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Summary")
    st.dataframe(input_df)

with col2:
    st.subheader("Prediction Result")

    if st.button("Predict Cancer Stage") and name.strip() != "":

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        stage = get_stage(probability)
        clinical_stage = calculate_stage(tumor_size, lymph_nodes, metastasis)

        st.progress(float(probability))
        st.write("### AI Predicted Stage:", stage)
        st.write("### TNM Clinical Stage:", clinical_stage)
        st.write(f"Risk Probability: {probability:.2%}")

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie([1 - probability, probability], labels=["No Cancer", "Cancer"], autopct="%1.1f%%")
        st.pyplot(fig)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text': "Cancer Risk %"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig_gauge)

        # ROC
        y_prob = [0, probability]
        y_true = [0, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0,1],[0,1],'--')
        ax2.legend()
        st.pyplot(fig2)

        # SHAP
        st.subheader("Model Explanation (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        plt.figure()
        shap.summary_plot(shap_values, input_df, show=False)
        st.pyplot(plt.gcf())

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": model.feature_names_in_,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)
            st.bar_chart(importance_df.set_index("Feature"))

        # Stage Details
        info = stage_details(stage)
        st.subheader("Symptoms")
        st.write(info["Symptoms"])
        st.subheader("Diet Recommendation")
        st.write(info["Diet"])
        st.subheader("Precautions")
        st.write(info["Precautions"])

        # Save Record
        c.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (name, age, gender, stage, clinical_stage,
                   probability, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

        # PDF Download
        pdf_buffer = create_pdf(name, age, gender, stage, clinical_stage, probability)
        st.download_button(
            label="📥 Download Full Medical Report",
            data=pdf_buffer,
            file_name=f"{name}_Cancer_Report.pdf",
            mime="application/pdf"
        )

    elif name.strip() == "":
        st.warning("Please enter patient name before prediction.")

if st.button("View Past Records"):
    df_records = pd.read_sql("SELECT * FROM records", conn)
    st.dataframe(df_records)

st.markdown("---")
st.caption("Final Year Project | Priya Mothukuri | ML • SHAP • ROC • Clinical DSS")