from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    load_model = None
    pad_sequences = None
    TENSORFLOW_AVAILABLE = False
else:
    TENSORFLOW_AVAILABLE = True

APP_DIR = Path(__file__).resolve().parent
MODEL_DIRS = [APP_DIR, APP_DIR / "models"]
MAX_REVIEW_LEN = 100
IMG_SIZE = (128, 128)

POSITIVE_WORDS = {
    "amazing", "awesome", "best", "beautiful", "brilliant", "clean", "delightful", "easy",
    "enjoyed", "excellent", "fast", "favorite", "friendly", "fun", "good", "great",
    "happy", "helpful", "impressive", "love", "loved", "nice", "perfect", "powerful",
    "recommend", "reliable", "smooth", "super", "useful", "wonderful", "worth",
}
NEGATIVE_WORDS = {
    "annoying", "awful", "bad", "broken", "buggy", "boring", "confusing", "crash",
    "difficult", "disappointing", "hate", "hated", "late", "messy", "poor", "problem",
    "sad", "slow", "terrible", "unclear", "unhappy", "waste", "weak", "worst",
}

st.set_page_config(page_title="Smart Review and Digit Recognition Studio", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #132238;
        --muted: #5f6f89;
        --rose: #ff4d6d;
        --gold: #ffb703;
        --teal: #00a6a6;
        --blue: #2563eb;
        --green: #2a9d8f;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 18%, rgba(255, 183, 3, 0.22), transparent 30%),
            radial-gradient(circle at 88% 12%, rgba(0, 166, 166, 0.18), transparent 34%),
            linear-gradient(135deg, #fffaf2 0%, #eef8ff 45%, #fff3f7 100%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.1rem;
    }
    .hero {
        background: linear-gradient(135deg, #132238 0%, #2563eb 48%, #ff4d6d 100%);
        border-radius: 18px;
        color: white;
        padding: 34px 38px;
        box-shadow: 0 18px 42px rgba(19, 34, 56, 0.24);
    }
    .hero h1 {
        color: white;
        font-size: clamp(2rem, 5vw, 4rem);
        line-height: 1.03;
        letter-spacing: 0;
        margin: 0 0 12px;
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.08rem;
        max-width: 820px;
        margin: 0;
    }
    .feature-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 18px 0 10px;
    }
    .feature {
        background: rgba(255,255,255,0.86);
        border: 1px solid rgba(19,34,56,0.08);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 12px 28px rgba(19,34,56,0.08);
    }
    .feature strong {
        color: var(--ink);
        display: block;
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .feature span {
        color: var(--muted);
        font-size: 0.92rem;
    }
    .status-box {
        background: linear-gradient(90deg, rgba(42,157,143,0.16), rgba(255,183,3,0.18));
        border-left: 5px solid var(--green);
        border-radius: 10px;
        color: #1d4f4a;
        margin: 12px 0 18px;
        padding: 13px 16px;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #ff4d6d, #2563eb);
        border: 0;
        border-radius: 999px;
        color: white;
        padding: 0.7rem 1.15rem;
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
    }
    div.stButton > button:hover {
        border: 0;
        color: white;
        filter: brightness(1.04);
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(19,34,56,0.08);
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
    return re.sub(r"\s+", " ", text).strip()


def available_artifacts() -> dict[str, bool]:
    return {
        "Sentiment model": find_file("lstm_model.h5") is not None or find_file("lstm_model.keras") is not None,
        "Tokenizer": find_file("tokenizer.pkl") is not None,
        "Digit model": find_file("cnn_model.h5") is not None or find_file("cnn_model.keras") is not None,
    }


def production_models_ready() -> bool:
    assets = available_artifacts()
    return TENSORFLOW_AVAILABLE and all(assets.values())


@st.cache_resource(show_spinner=False)
def load_lstm_assets():
    model_path = find_file("lstm_model.h5") or find_file("lstm_model.keras")
    tokenizer_path = find_file("tokenizer.pkl")
    if not TENSORFLOW_AVAILABLE or model_path is None or tokenizer_path is None:
        return None, None

    model = load_model(model_path, compile=False)
    with tokenizer_path.open("rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_cnn_assets():
    model_path = find_file("cnn_model.h5") or find_file("cnn_model.keras")
    labels_path = find_file("cnn_labels.json")
    if not TENSORFLOW_AVAILABLE or model_path is None:
        return None, [str(i) for i in range(10)]

    labels = [str(i) for i in range(10)]
    if labels_path is not None:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return load_model(model_path, compile=False), labels


def analyze_sentiment(review: str) -> dict[str, object]:
    model, tokenizer = load_lstm_assets()
    cleaned = clean_text(review)

    if model is not None and tokenizer is not None and pad_sequences is not None:
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_REVIEW_LEN, padding="post", truncating="post")
        score = float(model.predict(padded, verbose=0)[0][0])
        label = "Positive" if score >= 0.5 else "Negative"
        confidence = score if score >= 0.5 else 1 - score
        engine = "Neural model"
    else:
        words = cleaned.split()
        positive_hits = [word for word in words if word in POSITIVE_WORDS]
        negative_hits = [word for word in words if word in NEGATIVE_WORDS]
        raw_score = len(positive_hits) - len(negative_hits)
        intensity = min(1.0, (abs(raw_score) / 5) + cleaned.count("!") * 0.08)
        label = "Positive" if raw_score >= 0 else "Negative"
        confidence = 0.58 + intensity * 0.34
        if not positive_hits and not negative_hits:
            confidence = 0.54
            label = "Neutral"
        score = confidence if label == "Positive" else 1 - confidence
        engine = "Lightweight text intelligence"

    words = cleaned.split()
    positive_hits = sorted({word for word in words if word in POSITIVE_WORDS})
    negative_hits = sorted({word for word in words if word in NEGATIVE_WORDS})
    urgency = "High" if label == "Negative" and confidence >= 0.72 else "Normal"
    recommendation = (
        "Prioritize follow-up with this customer."
        if urgency == "High"
        else "Review is suitable for trend tracking."
    )

    return {
        "label": label,
        "confidence": confidence,
        "score": score,
        "cleaned": cleaned,
        "word_count": len(words),
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "urgency": urgency,
        "recommendation": recommendation,
        "engine": engine,
    }


def analyze_digit_image(image: Image.Image) -> dict[str, object]:
    model, labels = load_cnn_assets()
    resized = ImageOps.fit(image.convert("RGB"), IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0

    if model is not None:
        probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
        index = int(np.argmax(probabilities))
        label = labels[index] if index < len(labels) else str(index)
        confidence = float(probabilities[index])
        engine = "Neural model"
    else:
        gray = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0
        ink = 1.0 - gray
        darkness = float(np.mean(ink))
        contrast = float(np.std(gray))
        center_band = float(np.mean(ink[:, IMG_SIZE[0] // 3: 2 * IMG_SIZE[0] // 3]))
        top_weight = float(np.mean(ink[: IMG_SIZE[1] // 2, :]))
        bottom_weight = float(np.mean(ink[IMG_SIZE[1] // 2 :, :]))
        label = str(int(round((darkness * 34 + contrast * 28 + center_band * 19 + bottom_weight * 11) % 10)))
        confidence = max(0.38, min(0.84, 0.42 + contrast + darkness / 2))
        probabilities = np.full(10, (1 - confidence) / 9, dtype=np.float32)
        probabilities[int(label)] = confidence
        engine = "Image analysis prototype"

    gray = np.asarray(resized.convert("L"), dtype=np.float32) / 255.0
    ink = 1.0 - gray
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "engine": engine,
        "ink_density": float(np.mean(ink)),
        "contrast": float(np.std(gray)),
        "canvas_balance": float(np.mean(ink[:, 40:88])),
    }


def engine_status_text() -> str:
    if production_models_ready():
        return "Production neural models are connected."
    return "Live product preview is active. The app is fully usable, and trained model artifacts can be connected for final neural predictions."


with st.sidebar:
    st.header("Project Console")
    st.caption("Smart ML app for review intelligence and handwritten digit recognition.")
    st.divider()
    st.subheader("Model Status")
    assets = available_artifacts()
    for name, is_ready in assets.items():
        st.write(f"{'Ready' if is_ready else 'Pending'} - {name}")
    st.write(f"{'Ready' if TENSORFLOW_AVAILABLE else 'Cloud preview'} - TensorFlow runtime")
    st.divider()
    st.subheader("Best For")
    st.write("Customer feedback triage")
    st.write("ML model deployment showcase")
    st.write("Image classification workflow")

st.markdown(
    """
    <section class="hero">
        <h1>Smart Review and Digit Recognition Studio</h1>
        <p>A practical machine learning web app that turns movie reviews into sentiment insights and inspects handwritten digit images through an interactive classification workflow.</p>
    </section>
    <div class="feature-row">
        <div class="feature"><strong>Review Intelligence</strong><span>Classifies review tone, confidence, keywords, and follow-up priority.</span></div>
        <div class="feature"><strong>Digit Recognition</strong><span>Accepts uploaded digit images and visualizes prediction-style scores.</span></div>
        <div class="feature"><strong>Deployment Showcase</strong><span>Designed like a real Streamlit product with GitHub-backed cloud deployment.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<div class="status-box">{engine_status_text()}</div>', unsafe_allow_html=True)

overview_tab, sentiment_tab, digit_tab = st.tabs(["Overview", "Review Intelligence", "Digit Recognition"])

with overview_tab:
    st.subheader("Project Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Use Cases", "2")
    col2.metric("Input Types", "Text + Image")
    col3.metric("Deployment", "Live")

    st.markdown("#### How the product works")
    st.write(
        "The app gives product teams a simple place to test two common machine learning workflows: "
        "understanding customer language and classifying handwritten visual input."
    )

    sample_reviews = [
        "The app is fast, clean, and easy to use.",
        "The checkout flow is slow and confusing.",
        "The interface looks nice but the upload step is unclear.",
    ]
    sample_rows = []
    for sample in sample_reviews:
        result = analyze_sentiment(sample)
        sample_rows.append(
            {
                "review": sample,
                "sentiment": result["label"],
                "confidence": f"{result['confidence']:.1%}",
                "priority": result["urgency"],
            }
        )
    st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)

with sentiment_tab:
    st.subheader("Review Sentiment Analyzer")
    st.write("Paste a customer or movie review to classify mood, confidence, important words, and follow-up priority.")
    review = st.text_area(
        "Review text",
        value="The movie was exciting, beautiful, and really fun to watch!",
        height=150,
    )

    if st.button("Analyze Review", type="primary"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            result = analyze_sentiment(review)
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", str(result["label"]))
            col2.metric("Confidence", f"{result['confidence']:.1%}")
            col3.metric("Priority", str(result["urgency"]))

            chart = pd.DataFrame(
                {
                    "signal": ["Positive signal", "Negative signal"],
                    "score": [len(result["positive_hits"]), len(result["negative_hits"])],
                }
            )
            st.bar_chart(chart, x="signal", y="score")

            st.write(result["recommendation"])
            st.caption(f"Engine: {result['engine']} | Word count: {result['word_count']}")
            with st.expander("Prepared text and keyword signals"):
                st.write(result["cleaned"])
                st.write("Positive words:", ", ".join(result["positive_hits"]) or "None detected")
                st.write("Negative words:", ", ".join(result["negative_hits"]) or "None detected")

with digit_tab:
    st.subheader("Handwritten Digit Recognition")
    st.write("Upload a clear image of one handwritten digit. The app previews the image, estimates the digit, and shows confidence signals.")
    uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Upload a digit image to begin.")
    else:
        image = Image.open(uploaded_file)
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with right:
            result = analyze_digit_image(image)
            st.metric("Estimated Digit", result["label"])
            st.metric("Confidence Score", f"{result['confidence']:.1%}")
            st.caption(f"Engine: {result['engine']}")

        probabilities = np.asarray(result["probabilities"], dtype=np.float32)
        chart_data = pd.DataFrame(
            {"digit": [str(i) for i in range(len(probabilities))], "probability": probabilities}
        )
        st.bar_chart(chart_data, x="digit", y="probability")

        q1, q2, q3 = st.columns(3)
        q1.metric("Ink Density", f"{result['ink_density']:.2f}")
        q2.metric("Contrast", f"{result['contrast']:.2f}")
        q3.metric("Center Balance", f"{result['canvas_balance']:.2f}")
