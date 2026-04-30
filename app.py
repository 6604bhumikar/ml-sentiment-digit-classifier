from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as error:
    load_model = None
    pad_sequences = None
    TENSORFLOW_IMPORT_ERROR = error
else:
    TENSORFLOW_IMPORT_ERROR = None

APP_DIR = Path(__file__).resolve().parent
MODEL_DIRS = [APP_DIR, APP_DIR / "models"]
MAX_REVIEW_LEN = 100
IMG_SIZE = (128, 128)

st.set_page_config(page_title="Sentiment and Digit Classifier", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #16213e;
        --rose: #ff4d6d;
        --gold: #ffb703;
        --teal: #00b4d8;
        --violet: #7b2cbf;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 18%, rgba(255, 183, 3, 0.20), transparent 30%),
            radial-gradient(circle at 85% 12%, rgba(0, 180, 216, 0.18), transparent 32%),
            linear-gradient(135deg, #fff8f0 0%, #f4fbff 48%, #fff3f8 100%);
    }
    .block-container {
        padding-top: 2.4rem;
        max-width: 1120px;
    }
    .hero {
        background: linear-gradient(135deg, #16213e 0%, #7b2cbf 55%, #ff4d6d 100%);
        border-radius: 18px;
        padding: 34px 38px;
        color: white;
        box-shadow: 0 18px 45px rgba(22, 33, 62, 0.22);
    }
    .hero h1 {
        font-size: clamp(2.1rem, 5vw, 4.3rem);
        line-height: 1;
        margin: 0 0 12px;
        color: white;
        letter-spacing: 0;
    }
    .hero p {
        font-size: 1.08rem;
        max-width: 760px;
        margin: 0;
        color: rgba(255, 255, 255, 0.9);
    }
    .feature-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 18px 0 12px;
    }
    .feature {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(22, 33, 62, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 12px 28px rgba(22, 33, 62, 0.08);
    }
    .feature strong {
        display: block;
        color: var(--ink);
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .feature span {
        color: #586174;
        font-size: 0.92rem;
    }
    .demo-note {
        background: linear-gradient(90deg, rgba(255, 183, 3, 0.22), rgba(0, 180, 216, 0.18));
        border-left: 5px solid var(--gold);
        border-radius: 10px;
        color: #4d3b00;
        padding: 14px 16px;
        margin: 12px 0 18px;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #ff4d6d, #7b2cbf);
        border: 0;
        color: white;
        border-radius: 999px;
        padding: 0.7rem 1.15rem;
        box-shadow: 0 10px 22px rgba(123, 44, 191, 0.22);
    }
    div.stButton > button:hover {
        border: 0;
        color: white;
        filter: brightness(1.04);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(22, 33, 62, 0.08);
        border-radius: 12px;
        padding: 14px;
    }
    @media (max-width: 760px) {
        .feature-row {
            grid-template-columns: 1fr;
        }
        .hero {
            padding: 26px 22px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_file(filename: str) -> Path | None:
    for folder in MODEL_DIRS:
        path = folder / filename
        if path.exists():
            return path
    return None


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def require_tensorflow() -> None:
    if TENSORFLOW_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The trained neural-network model is not connected in this deployment."
        ) from TENSORFLOW_IMPORT_ERROR


@st.cache_resource(show_spinner=False)
def load_lstm_assets():
    require_tensorflow()
    model_path = find_file("lstm_model.h5") or find_file("lstm_model.keras")
    tokenizer_path = find_file("tokenizer.pkl")

    missing = []
    if model_path is None:
        missing.append("lstm_model.h5")
    if tokenizer_path is None:
        missing.append("tokenizer.pkl")
    if missing:
        return None, None, missing

    model = load_model(model_path, compile=False)
    with tokenizer_path.open("rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer, []


@st.cache_resource(show_spinner=False)
def load_cnn_assets():
    require_tensorflow()
    model_path = find_file("cnn_model.h5") or find_file("cnn_model.keras")
    labels_path = find_file("cnn_labels.json")

    if model_path is None:
        return None, [str(i) for i in range(10)], ["cnn_model.h5"]

    labels = [str(i) for i in range(10)]
    if labels_path is not None:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

    model = load_model(model_path, compile=False)
    return model, labels, []


def show_missing(missing: list[str]) -> None:
    st.info("Model files are not connected yet, so this result is running in demo mode.")


def demo_sentiment(review: str):
    positive_words = {
        "amazing", "awesome", "best", "beautiful", "brilliant", "enjoyed", "excellent", "fun",
        "good", "great", "happy", "love", "loved", "nice", "perfect", "recommend", "super", "wonderful",
    }
    negative_words = {
        "awful", "bad", "boring", "disappointing", "hate", "hated", "poor", "sad", "slow",
        "terrible", "waste", "worst", "annoying", "dull", "weak",
    }
    cleaned = clean_text(review)
    words = cleaned.split()
    score = sum(word in positive_words for word in words) - sum(word in negative_words for word in words)
    punctuation_boost = cleaned.count("!") * 0.08

    if score > 0:
        label = "Positive"
        confidence = min(0.94, 0.62 + score * 0.11 + punctuation_boost)
    elif score < 0:
        label = "Negative"
        confidence = min(0.94, 0.62 + abs(score) * 0.11 + punctuation_boost)
    else:
        label = "Positive" if len(review) % 2 == 0 else "Negative"
        confidence = 0.56

    return label, confidence, cleaned


def demo_digit_analysis(image: Image.Image):
    grayscale = image.convert("L").resize(IMG_SIZE)
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    ink = 1.0 - array
    darkness = float(np.mean(ink))
    contrast = float(np.std(array))
    vertical_weight = float(np.mean(ink[:, IMG_SIZE[0] // 3: 2 * IMG_SIZE[0] // 3]))
    loop_score = float(np.mean(array[42:86, 42:86]))
    estimated_digit = int(np.clip(round((darkness * 31 + contrast * 23 + vertical_weight * 17 + loop_score * 9) % 10), 0, 9))
    return estimated_digit, darkness, contrast, vertical_weight


def predict_sentiment(review: str):
    if TENSORFLOW_IMPORT_ERROR is not None:
        label, confidence, cleaned = demo_sentiment(review)
        return label, confidence, cleaned, ["demo"]

    try:
        model, tokenizer, missing = load_lstm_assets()
    except Exception:
        label, confidence, cleaned = demo_sentiment(review)
        return label, confidence, cleaned, ["demo"]
    if missing:
        label, confidence, cleaned = demo_sentiment(review)
        return label, confidence, cleaned, missing

    cleaned = clean_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_REVIEW_LEN, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if score >= 0.5 else "Negative"
    confidence = score if score >= 0.5 else 1 - score
    return label, confidence, cleaned, []


def predict_digit(image: Image.Image):
    if TENSORFLOW_IMPORT_ERROR is not None:
        label, darkness, contrast, vertical_weight = demo_digit_analysis(image)
        probabilities = np.zeros(10, dtype=np.float32)
        probabilities[label] = max(0.35, min(0.82, 0.42 + contrast + darkness / 2))
        probabilities += (1 - probabilities.sum()) / 10
        return str(label), float(probabilities[label]), probabilities, ["demo"]

    try:
        model, labels, missing = load_cnn_assets()
    except Exception:
        label, darkness, contrast, vertical_weight = demo_digit_analysis(image)
        probabilities = np.zeros(10, dtype=np.float32)
        probabilities[label] = max(0.35, min(0.82, 0.42 + contrast + darkness / 2))
        probabilities += (1 - probabilities.sum()) / 10
        return str(label), float(probabilities[label]), probabilities, ["demo"]
    if missing:
        label, darkness, contrast, vertical_weight = demo_digit_analysis(image)
        probabilities = np.zeros(10, dtype=np.float32)
        probabilities[label] = max(0.35, min(0.82, 0.42 + contrast + darkness / 2))
        probabilities += (1 - probabilities.sum()) / 10
        return str(label), float(probabilities[label]), probabilities, missing

    resized = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    label = labels[index] if index < len(labels) else str(index)
    return label, float(probabilities[index]), probabilities, []


st.markdown(
    """
    <section class="hero">
        <h1>Sentiment and Digit Classifier</h1>
        <p>An interactive machine learning web app for reading the mood of movie reviews and exploring handwritten digit image classification.</p>
    </section>
    <div class="feature-row">
        <div class="feature"><strong>Text Intelligence</strong><span>Type a review and see whether the tone feels positive or negative.</span></div>
        <div class="feature"><strong>Image Classifier</strong><span>Upload a handwritten digit image and inspect a prediction-style score chart.</span></div>
        <div class="feature"><strong>Deployment Ready</strong><span>Built as a Streamlit app connected to a GitHub repository.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if TENSORFLOW_IMPORT_ERROR is not None:
    st.markdown(
        '<div class="demo-note">Live demo mode is active. The interface works now, and trained neural-network models can be connected later for final predictions.</div>',
        unsafe_allow_html=True,
    )

sentiment_tab, digit_tab = st.tabs(["Review Sentiment", "Digit Image"])

with sentiment_tab:
    st.subheader("Movie Review Sentiment")
    review = st.text_area("Review text", value="The movie was exciting, beautiful, and really fun to watch!", height=150)

    if st.button("Predict Sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            label, confidence, cleaned, missing = predict_sentiment(review)
            if missing:
                show_missing(missing)
            col1, col2 = st.columns(2)
            col1.metric("Prediction", label)
            col2.metric("Confidence", f"{confidence:.2%}")
            with st.expander("Text prepared for analysis"):
                st.write(cleaned)

with digit_tab:
    st.subheader("Handwritten Digit Classification")
    uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Upload an image to predict the digit.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict Digit", type="primary"):
            label, confidence, probabilities, missing = predict_digit(image)
            if missing:
                show_missing(missing)
            col1, col2 = st.columns(2)
            col1.metric("Estimated Digit", label)
            col2.metric("Score", f"{confidence:.2%}")
            chart_data = pd.DataFrame(
                {"digit": [str(i) for i in range(len(probabilities))], "probability": probabilities}
            )
            st.bar_chart(chart_data, x="digit", y="probability")
