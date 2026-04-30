from __future__ import annotations

import base64
import io
import re
import string
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SENTIMENT_SAMPLES = [
    ("I loved the product and the support team was excellent", "Positive"),
    ("The movie was beautiful, emotional, and very enjoyable", "Positive"),
    ("This app is fast, clean, useful, and easy to understand", "Positive"),
    ("The delivery was quick and the quality was amazing", "Positive"),
    ("I am happy with the service and would recommend it", "Positive"),
    ("The new dashboard is smooth and the insights are helpful", "Positive"),
    ("Great experience, friendly staff, and reliable performance", "Positive"),
    ("The phone camera is brilliant and the battery is perfect", "Positive"),
    ("I had a wonderful experience using this tool", "Positive"),
    ("Everything worked as expected and the result was impressive", "Positive"),
    ("The interface is nice but the loading time is average", "Neutral"),
    ("The product arrived today and I have not tested it yet", "Neutral"),
    ("It is okay, neither great nor terrible", "Neutral"),
    ("The movie had some good scenes and some boring parts", "Neutral"),
    ("The service was acceptable but nothing special", "Neutral"),
    ("The app has useful features but needs better documentation", "Neutral"),
    ("The design is simple and the performance is normal", "Neutral"),
    ("I received the update and will check it later", "Neutral"),
    ("The experience was mixed and I need more time to decide", "Neutral"),
    ("The order status changed but I do not have an opinion yet", "Neutral"),
    ("The product is terrible and completely disappointing", "Negative"),
    ("I hated the movie because it was boring and slow", "Negative"),
    ("The website crashed and the checkout process was awful", "Negative"),
    ("Customer service was rude and the problem was not fixed", "Negative"),
    ("This is the worst experience I have had with an app", "Negative"),
    ("The delivery was late and the packaging was damaged", "Negative"),
    ("The interface is confusing and full of bugs", "Negative"),
    ("The model gave poor results and wasted my time", "Negative"),
    ("I am unhappy with the quality and want a refund", "Negative"),
    ("The tool is unreliable, slow, and difficult to use", "Negative"),
    ("Excellent analytics helped our brand respond faster", "Positive"),
    ("The latest version made our workflow easier", "Positive"),
    ("Support solved my issue quickly and politely", "Positive"),
    ("The report is clear and valuable for the team", "Positive"),
    ("Some pages are useful while others feel unfinished", "Neutral"),
    ("The notification arrived but the context was unclear", "Neutral"),
    ("The trial gives enough information to evaluate it", "Neutral"),
    ("The signup flow failed and the error message was confusing", "Negative"),
    ("The dashboard is slow and the charts keep freezing", "Negative"),
    ("I cannot recommend this because it breaks too often", "Negative"),
]


class SentimentClassifier:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_path = model_dir / "sentiment_pipeline.joblib"
        self.meta_path = model_dir / "sentiment_meta.joblib"
        self.pipeline: Pipeline | None = None
        self.meta: dict[str, object] = {}

    def ensure_model(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.pipeline is not None:
            return
        if self.model_path.exists() and self.meta_path.exists():
            self.pipeline = joblib.load(self.model_path)
            self.meta = joblib.load(self.meta_path)
            return
        self.train()

    def train(self) -> None:
        texts = [text for text, _ in SENTIMENT_SAMPLES]
        labels = [label for _, label in SENTIMENT_SAMPLES]
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=preprocess_text,
                        stop_words=sorted(ENGLISH_STOP_WORDS),
                        ngram_range=(1, 2),
                        min_df=1,
                    ),
                ),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels,
        )
        pipeline.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, pipeline.predict(x_test))
        self.pipeline = pipeline
        self.meta = {
            "algorithm": "TF-IDF + Logistic Regression",
            "training_examples": len(SENTIMENT_SAMPLES),
            "validation_accuracy": round(float(accuracy), 3),
        }
        joblib.dump(pipeline, self.model_path)
        joblib.dump(self.meta, self.meta_path)

    def predict(self, text: str) -> dict[str, object]:
        self.ensure_model()
        assert self.pipeline is not None
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        best_index = int(np.argmax(probabilities))
        label = str(classes[best_index])
        confidence = round(float(probabilities[best_index]) * 100, 2)
        probability_map = {
            str(class_name): round(float(probability) * 100, 2)
            for class_name, probability in zip(classes, probabilities)
        }
        explanation = self.explain(text, label)
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probability_map,
            "processed_text": preprocess_text(text),
            "explanation": explanation,
            "model": self.meta,
        }

    def explain(self, text: str, label: str) -> str:
        assert self.pipeline is not None
        vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        features = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        class_index = list(classifier.classes_).index(label)
        contributions = features.multiply(classifier.coef_[class_index]).toarray()[0]
        top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
        top_terms = [feature_names[index] for index in top_indices if contributions[index] != 0]
        if not top_terms:
            return "The prediction is based on the overall TF-IDF pattern after preprocessing."
        return "Most influential text signals: " + ", ".join(top_terms)

    def status(self) -> dict[str, object]:
        self.ensure_model()
        return {"ready": self.pipeline is not None, **self.meta}


class DigitClassifier:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_path = model_dir / "digit_classifier.joblib"
        self.meta_path = model_dir / "digit_meta.joblib"
        self.model: Pipeline | None = None
        self.meta: dict[str, object] = {}

    def ensure_model(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            return
        if self.model_path.exists() and self.meta_path.exists():
            self.model = joblib.load(self.model_path)
            self.meta = joblib.load(self.meta_path)
            return
        self.train()

    def train(self) -> None:
        digits = load_digits()
        x_train, x_test, y_train, y_test = train_test_split(
            digits.data,
            digits.target,
            test_size=0.22,
            random_state=42,
            stratify=digits.target,
        )
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2500, solver="lbfgs")),
            ]
        )
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        self.model = model
        self.meta = {
            "algorithm": "Scikit-learn digits + Logistic Regression",
            "training_examples": int(len(x_train)),
            "validation_accuracy": round(float(accuracy), 3),
        }
        joblib.dump(model, self.model_path)
        joblib.dump(self.meta, self.meta_path)

    def predict(self, image_data: str) -> dict[str, object]:
        self.ensure_model()
        assert self.model is not None
        features, preview_stats = image_to_digit_features(image_data)
        probabilities = self.model.predict_proba([features])[0]
        best_index = int(np.argmax(probabilities))
        label = str(self.model.classes_[best_index])
        confidence = round(float(probabilities[best_index]) * 100, 2)
        probability_map = {
            str(class_name): round(float(probability) * 100, 2)
            for class_name, probability in zip(self.model.classes_, probabilities)
        }
        explanation = (
            f"The classifier resized the image to an 8x8 grayscale grid. "
            f"Ink density is {preview_stats['ink_density']:.2f}, contrast is {preview_stats['contrast']:.2f}, "
            f"and the strongest class probability is digit {label}."
        )
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probability_map,
            "features": preview_stats,
            "explanation": explanation,
            "model": self.meta,
        }

    def status(self) -> dict[str, object]:
        self.ensure_model()
        return {"ready": self.model is not None, **self.meta}


def preprocess_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"<[^>]+>", " ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in lowered.split() if token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def image_to_digit_features(image_data: str) -> tuple[np.ndarray, dict[str, float]]:
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    raw = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(raw)).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (8, 8), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 0:
        array = array / array.max() * 16.0
    features = array.reshape(-1)
    stats = {
        "ink_density": round(float(np.mean(array) / 16.0), 3),
        "contrast": round(float(np.std(array) / 16.0), 3),
        "center_weight": round(float(np.mean(array[2:6, 2:6]) / 16.0), 3),
    }
    return features, stats
