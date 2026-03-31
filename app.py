import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data_utils import load_data
from src.predict import predict_single, load_pipeline
from src.config import METRICS_PATH, TARGET_COLUMN

st.set_page_config(page_title="Credit Risk Prediction App", layout="wide")


@st.cache_resource
def get_model():
    return load_pipeline()


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


st.title("Credit Risk Prediction App")
st.write("An end-to-end machine learning project for banking-style credit risk assessment.")

try:
    model = get_model()
    df = get_data()
    metrics_payload = get_metrics()
except Exception as e:
    st.error(f"Failed to load app resources: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Overview", "Data Explorer", "Predict Applicant"])

with tab1:
    st.subheader("Project Overview")
    st.write("This app predicts whether an applicant is likely to be a good or bad credit risk.")
    st.write(f"Best selected model: **{metrics_payload['best_model']}**")
    st.write(f"Model selection metric: **{metrics_payload['selection_metric']}**")

    results = metrics_payload["results"]
    rows = []

    for model_name, vals in results.items():
        rows.append({
            "model": model_name,
            "accuracy": vals["accuracy"],
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1_score": vals["f1_score"],
            "roc_auc": vals["roc_auc"]
        })

    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True)

with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Target Distribution")
    target_counts = df[TARGET_COLUMN].value_counts()

    fig, ax = plt.subplots()
    target_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Credit Risk")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    st.pyplot(fig)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with tab3:
    st.subheader("Predict Applicant Risk")

    col1, col2 = st.columns(2)

    with col1:
        status = st.selectbox("Status", sorted(df["status"].dropna().astype(str).unique()))
        duration = st.number_input("Duration (months)", min_value=1, max_value=120, value=12)
        credit_history = st.selectbox("Credit History", sorted(df["credit_history"].dropna().astype(str).unique()))
        purpose = st.selectbox("Purpose", sorted(df["purpose"].dropna().astype(str).unique()))
        amount = st.number_input("Credit Amount", min_value=0, value=5000)
        savings = st.selectbox("Savings", sorted(df["savings"].dropna().astype(str).unique()))
        employment_duration = st.selectbox(
            "Employment Duration",
            sorted(df["employment_duration"].dropna().astype(str).unique())
        )
        installment_rate = st.selectbox(
            "Installment Rate",
            sorted(df["installment_rate"].dropna().astype(str).unique())
        )
        personal_status_sex = st.selectbox(
            "Personal Status / Sex",
            sorted(df["personal_status_sex"].dropna().astype(str).unique())
        )
        other_debtors = st.selectbox(
            "Other Debtors",
            sorted(df["other_debtors"].dropna().astype(str).unique())
        )

    with col2:
        present_residence = st.selectbox(
            "Present Residence",
            sorted(df["present_residence"].dropna().astype(str).unique())
        )
        property_type = st.selectbox("Property", sorted(df["property"].dropna().astype(str).unique()))
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        other_installment_plans = st.selectbox(
            "Other Installment Plans",
            sorted(df["other_installment_plans"].dropna().astype(str).unique())
        )
        housing = st.selectbox("Housing", sorted(df["housing"].dropna().astype(str).unique()))
        number_credits = st.selectbox(
            "Number of Existing Credits",
            sorted(df["number_credits"].dropna().astype(str).unique())
        )
        job = st.selectbox("Job", sorted(df["job"].dropna().astype(str).unique()))
        people_liable = st.selectbox(
            "People Liable",
            sorted(df["people_liable"].dropna().astype(str).unique())
        )
        telephone = st.selectbox("Telephone", sorted(df["telephone"].dropna().astype(str).unique()))
        foreign_worker = st.selectbox(
            "Foreign Worker",
            sorted(df["foreign_worker"].dropna().astype(str).unique())
        )

    if st.button("Predict Risk"):
        input_data = {
            "status": status,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "amount": amount,
            "savings": savings,
            "employment_duration": employment_duration,
            "installment_rate": installment_rate,
            "personal_status_sex": personal_status_sex,
            "other_debtors": other_debtors,
            "present_residence": present_residence,
            "property": property_type,
            "age": age,
            "other_installment_plans": other_installment_plans,
            "housing": housing,
            "number_credits": number_credits,
            "job": job,
            "people_liable": people_liable,
            "telephone": telephone,
            "foreign_worker": foreign_worker
        }

        result = predict_single(input_data)
        pred = result["prediction"]
        prob = result["probability_bad_risk"]

        st.subheader("Prediction Result")

        if str(pred).lower() == "bad":
            st.error("Predicted: Bad Risk")
        else:
            st.success("Predicted: Good Risk")

        st.write(f"Probability of bad risk: **{prob:.2%}**")

        if prob > 0.7:
            st.write("Interpretation: The model is strongly leaning toward higher credit risk.")
        elif prob > 0.4:
            st.write("Interpretation: The model sees moderate credit risk.")
        else:
            st.write("Interpretation: The model sees relatively lower credit risk.")