from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from ml_models import DigitClassifier, SentimentClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="ML Sentiment & Digit Classifier",
    page_icon=":material/psychology:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #19283d 0, #0b1119 38%, #06090f 100%);
        color: #e8eef7;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101827 0%, #080d14 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    .hero {
        padding: 30px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(15, 23, 42, 0.90));
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
        margin-bottom: 22px;
    }
    .hero h1 {
        color: #f8fafc;
        font-size: 2.15rem;
        margin: 0 0 8px;
        letter-spacing: 0;
    }
    .hero p {
        color: #b9c7d8;
        font-size: 1.02rem;
        margin: 0;
        max-width: 880px;
    }
    .metric-card {
        padding: 16px 18px;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.72);
    }
    .metric-label {
        color: #93a4b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .section-note {
        color: #9fb0c5;
        font-size: 0.92rem;
        margin-top: -8px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training sentiment and digit models...")
def load_models() -> tuple[SentimentClassifier, DigitClassifier]:
    sentiment_model = SentimentClassifier(MODEL_DIR)
    digit_model = DigitClassifier(MODEL_DIR)
    sentiment_model.ensure_model()
    digit_model.ensure_model()
    return sentiment_model, digit_model


def probability_table(probabilities: dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            [{"Class": label, "Probability (%)": probability} for label, probability in probabilities.items()]
        )
        .sort_values("Probability (%)", ascending=False)
        .reset_index(drop=True)
    )


def encode_uploaded_image(uploaded_file) -> str:
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{uploaded_file.type};base64,{encoded}"


sentiment_model, digit_model = load_models()

st.markdown(
    """
    <div class="hero">
        <h1>ML Sentiment & Handwritten Digit Classifier</h1>
        <p>A Streamlit machine learning dashboard for real-time text sentiment prediction and handwritten digit recognition using scikit-learn models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Console")
    st.success("Models loaded")
    st.markdown("**Sentiment model:** TF-IDF + Logistic Regression")
    st.markdown("**Digit model:** 8x8 image features + Logistic Regression")
    st.divider()
    st.caption("The models train automatically on first run and cache their artifacts in the app session.")

sentiment_status = sentiment_model.status()
digit_status = digit_model.status()

metric_cols = st.columns(4)
metrics = [
    ("Sentiment Accuracy", sentiment_status.get("validation_accuracy", "ready")),
    ("Sentiment Samples", sentiment_status.get("training_examples", "-")),
    ("Digit Accuracy", digit_status.get("validation_accuracy", "ready")),
    ("Digit Samples", digit_status.get("training_examples", "-")),
]
for col, (label, value) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

tab_sentiment, tab_digit = st.tabs(["Sentiment Analysis", "Digit Recognition"])

with tab_sentiment:
    st.subheader("Real-Time Sentiment Analysis")
    st.markdown(
        '<div class="section-note">Enter a review, comment, or customer message to classify it as Positive, Neutral, or Negative.</div>',
        unsafe_allow_html=True,
    )
    sample_text = st.text_area(
        "Text input",
        value="The dashboard is clean, fast, and very useful for understanding results.",
        height=140,
    )
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not sample_text.strip():
            st.warning("Please enter text before analyzing sentiment.")
        else:
            result = sentiment_model.predict(sample_text.strip())
            result_col, chart_col = st.columns([0.9, 1.1])
            with result_col:
                st.metric("Prediction", result["label"], f"{result['confidence']}% confidence")
                st.write(result["explanation"])
                st.caption("Processed text")
                st.code(result["processed_text"] or "No tokens after preprocessing", language="text")
            with chart_col:
                probabilities = probability_table(result["probabilities"])
                st.bar_chart(probabilities, x="Class", y="Probability (%)", use_container_width=True)
                st.dataframe(probabilities, hide_index=True, use_container_width=True)

with tab_digit:
    st.subheader("Handwritten Digit Recognition")
    st.markdown(
        '<div class="section-note">Upload a clear image of a single handwritten digit from 0 to 9. The app resizes it into the same 8x8 format used by scikit-learn digits.</div>',
        unsafe_allow_html=True,
    )
    uploaded_digit = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])
    if uploaded_digit:
        image_data = encode_uploaded_image(uploaded_digit)
        preview_col, output_col = st.columns([0.75, 1.25])
        with preview_col:
            st.image(uploaded_digit, caption="Uploaded digit", use_container_width=True)
        with output_col:
            if st.button("Classify Digit", type="primary", use_container_width=True):
                result = digit_model.predict(image_data)
                st.metric("Predicted Digit", result["label"], f"{result['confidence']}% confidence")
                st.write(result["explanation"])
                probabilities = probability_table(result["probabilities"])
                st.bar_chart(probabilities, x="Class", y="Probability (%)", use_container_width=True)
                st.dataframe(probabilities, hide_index=True, use_container_width=True)
                st.caption("Image feature summary")
                st.json(result["features"])
    else:
        st.info("Upload a digit image to run the classifier.")
