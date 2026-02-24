import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("cancer_model.pkl")

model = load_model()

# ---------------- PDF GENERATOR ---------------- #
def create_pdf(age, gender, prediction, probability):

    file_path = "Cancer_Report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Breast Cancer Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
    elements.append(Paragraph(f"Probability: {probability:.2%}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    if prediction == 1:
        result = "Cancer Detected"
        advice = [
            "Consult oncologist immediately",
            "Schedule biopsy test",
            "Perform imaging scans",
            "Do not delay diagnosis"
        ]
    else:
        result = "No Cancer Detected"
        advice = [
            "Routine screening",
            "Healthy lifestyle",
            "Annual checkup",
            "Monitor symptoms"
        ]

    elements.append(Paragraph(f"Result: {result}", styles["Heading2"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Recommendations:", styles["Heading3"]))

    elements.append(
        ListFlowable(
            [ListItem(Paragraph(item, styles["Normal"])) for item in advice]
        )
    )

    doc.build(elements)
    return file_path


# ---------------- HEADER ---------------- #
st.title("🩺 Breast Cancer Prediction System")
st.markdown("AI-Based Tumor Analysis using Machine Learning")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 1, 120, 30)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

st.sidebar.header("Tumor Measurements")

features = model.feature_names_in_
input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature.replace("_", " ").title(),
        value=0.0
    )

input_df = pd.DataFrame([input_data])

# ---------------- MAIN LAYOUT ---------------- #
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Input Summary")
    st.dataframe(input_df)

with col2:
    st.subheader("Prediction Result")

    if st.button("Predict Cancer Risk"):

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Progress Bar
        st.progress(float(probability))

        if prediction == 1:
            st.error(f"⚠ Cancer Detected\nProbability: {probability:.2%}")
        else:
            st.success(f"✅ No Cancer Detected\nProbability: {probability:.2%}")

        # Pie Chart
        st.subheader("Probability Distribution")

        labels = ["No Cancer", "Cancer"]
        values = [1 - probability, probability]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%")
        st.pyplot(fig)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            st.subheader("Top Feature Importance")

            importance_df = pd.DataFrame({
                "Feature": model.feature_names_in_,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)

            st.bar_chart(importance_df.set_index("Feature"))

        # Patient Info
        st.write("### Patient Information")
        st.write("Age:", age)
        st.write("Gender:", gender)

        # Recommendation
        st.subheader("Clinical Recommendation")

        if prediction == 1:
            st.warning("""
High risk detected.

Recommended actions:
• Consult oncologist immediately  
• Schedule biopsy  
• Imaging test required  
""")
        else:
            st.info("""
Low risk detected.

Recommended actions:
• Routine screening  
• Healthy lifestyle  
• Annual checkup  
""")

        # Download PDF
        pdf_path = create_pdf(age, gender, prediction, probability)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="Cancer_Report.pdf"
            )

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Final Year Project | Priya Mothukuri | Machine Learning • Streamlit • Python")