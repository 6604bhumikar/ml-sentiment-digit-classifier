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

st.set_page_config(page_title="Week 12 ML Deployment", layout="centered")


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
            "TensorFlow could not be imported. Install requirements with: pip install -r requirements.txt"
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
    st.error("Missing file(s): " + ", ".join(missing))
    st.info("Put tokenizer.pkl, lstm_model.h5, and cnn_model.h5 in this folder or inside a models folder.")


def predict_sentiment(review: str):
    model, tokenizer, missing = load_lstm_assets()
    if missing:
        return None, None, None, missing

    cleaned = clean_text(review)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_REVIEW_LEN, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if score >= 0.5 else "Negative"
    confidence = score if score >= 0.5 else 1 - score
    return label, confidence, cleaned, []


def predict_digit(image: Image.Image):
    model, labels, missing = load_cnn_assets()
    if missing:
        return None, None, None, missing

    resized = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    label = labels[index] if index < len(labels) else str(index)
    return label, float(probabilities[index]), probabilities, []


st.title("Week 12 ML Deployment")
st.caption("IMDB sentiment prediction with LSTM and handwritten digit prediction with CNN.")

if TENSORFLOW_IMPORT_ERROR is not None:
    st.warning("TensorFlow is not available in this environment yet. Install requirements before running predictions.")

sentiment_tab, digit_tab = st.tabs(["Sentiment", "Digit CNN"])

with sentiment_tab:
    st.subheader("Movie Review Sentiment")
    review = st.text_area("Review text", value="Movie is very good to watch.", height=150)

    if st.button("Predict Sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            try:
                label, confidence, cleaned, missing = predict_sentiment(review)
                if missing:
                    show_missing(missing)
                else:
                    st.metric("Prediction", label, f"{confidence:.2%} confidence")
                    with st.expander("Cleaned review"):
                        st.write(cleaned)
            except Exception as error:
                st.exception(error)

with digit_tab:
    st.subheader("Handwritten Digit Classification")
    uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Upload an image to predict the digit.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Predict Digit", type="primary"):
            try:
                label, confidence, probabilities, missing = predict_digit(image)
                if missing:
                    show_missing(missing)
                else:
                    st.metric("Prediction", label, f"{confidence:.2%} confidence")
                    chart_data = pd.DataFrame(
                        {"digit": [str(i) for i in range(len(probabilities))], "probability": probabilities}
                    )
                    st.bar_chart(chart_data, x="digit", y="probability")
            except Exception as error:
                st.exception(error)
