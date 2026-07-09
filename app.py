"""
Heart Disease Risk Predictor — Streamlit demo
Trains a Random Forest on the UCI heart disease dataset (heart.csv, must sit
next to this file) and lets a user enter their own stats to get a live
prediction. Random Forest was chosen because it was the best-performing model
in the project's comparison of 7 algorithms (see README.md).
"""

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")


@st.cache_resource
def load_model():
    dataset = pd.read_csv("heart.csv")
    X = dataset.drop("target", axis=1)
    y = dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    return model, list(X.columns), test_accuracy


st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "Enter your health information below to estimate heart disease risk. "
    "This is a **student project demo** using a Random Forest model trained "
    "on the UCI Heart Disease dataset (303 patients) — it is **not a medical "
    "diagnostic tool**, and should not be used to make real health decisions."
)

try:
    model, feature_order, test_accuracy = load_model()
except FileNotFoundError:
    st.error(
        "`heart.csv` not found. Place the UCI heart disease dataset "
        "(`heart.csv`) in the same folder as this app before running it."
    )
    st.stop()

st.caption(f"Model: Random Forest · Test accuracy on held-out data: {test_accuracy:.2f}%")

st.divider()
st.subheader("Your information")

col1, col2 = st.columns([1,1])

with col1:
    age = st.slider("Age", 18, 90, 22)
    sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)],index = 0, format_func=lambda x: x[0])[1]
    cp = st.selectbox(
        "Chest pain type",
        options=[
            ("Chest pain during exercise or stress", 0),
            ("Unusual Chest Pain", 1),
            ("Chest Pain not related to the heart", 2),
            ("No chest pain", 3)
        ],
        format_func=lambda x: x[0],
        index = 3
    )[1]
    trestbps = st.slider("Resting blood pressure (mm Hg) (Normal: 90-120 mmHg)", 80, 200, 100)
    chol = st.slider("Serum cholesterol (mg/dl) (Normal: Below 200 mg/dL)", 100, 600, 125)
    fbs = st.selectbox(
        "Fasting blood sugar > 120 mg/dl (High)?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0]
    )[1]
    thalach = st.slider("Max heart rate achieved during exercise or stress", 60, 220, 150)
    exang = st.selectbox(
        "Exercise-induced angina (Chest Pain)?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0]
    )[1]
with col2:
 st.subheader("Advanced Medical Information")
 st.caption(
    "These values come from specialized heart tests. "
    "If you've never had these tests, you can leave the defaults."
)
 with st.expander("🩺 Advanced Medical Test Results (Optional)"):
    restecg = st.selectbox(
        "Resting ECG results",
        options=[("Normal", 0), ("ST-T abnormality", 1), ("Left ventricular hypertrophy", 2)],
        format_func=lambda x: x[0],
    )[1]
    oldpeak = st.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox(
        "ECG change during exercise",
        options=[("Rising Pattern", 0), ("Flat", 1), ("Falling Pattern", 2)],
        format_func=lambda x: x[0],
            help="""
            Measures changes in your ECG during exercise.
            Doctors use this during a treadmill stress test. 
            If you've never had this test, leave the default value.
         """
    )[1]
    ca = st.selectbox("Number of major blood vessels seen on a heart scan (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox(
        "Heart stress scan result (Thallium Stress Test) ",
        options=[("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)],
        format_func=lambda x: x[0],
    )[1]

st.divider()

if st.button("Predict my risk", type="primary", use_container_width=True):
    input_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    input_df = pd.DataFrame([input_dict])[feature_order]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 0:
        st.error(f"⚠️ **Higher than average risk indicated** — model confidence: {(1-probability)*100:.1f}%")
    else:
        st.success(f"✅ **Lower than average risk indicated** — model confidence: {(probability)*100:.1f}%")

    st.caption(
        "Reminder: this reflects patterns in a 303-patient dataset, not personalized medical "
        "advice. Please consult a healthcare professional for an actual risk assessment."
    )

st.divider()
st.caption(
    "Built by Apan Sirohi (VIT Bhopal) · [Source on GitHub](https://github.com/apansirohi) · "
    "Model: Random Forest, 86.89% test accuracy"
)
