from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from database import HistoryStore
from ml_models import DigitClassifier, SentimentClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)
sentiment_model = SentimentClassifier(MODEL_DIR)
digit_model = DigitClassifier(MODEL_DIR)
history = HistoryStore(DATA_DIR / "predictions.sqlite3")


@app.before_request
def ensure_ready() -> None:
    sentiment_model.ensure_model()
    digit_model.ensure_model()
    history.ensure_database()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/sentiment")
def predict_sentiment():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Text is required."}), 400

    result = sentiment_model.predict(text)
    saved = history.add(
        task="sentiment",
        input_summary=text[:180],
        prediction=result["label"],
        confidence=result["confidence"],
        explanation=result["explanation"],
    )
    result["history_id"] = saved
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    return jsonify(result)


@app.post("/api/digit")
def predict_digit():
    payload = request.get_json(silent=True) or {}
    image_data = str(payload.get("image", "")).strip()
    if not image_data:
        return jsonify({"error": "Image data is required."}), 400

    result = digit_model.predict(image_data)
    saved = history.add(
        task="digit",
        input_summary="Uploaded or drawn digit image",
        prediction=str(result["label"]),
        confidence=result["confidence"],
        explanation=result["explanation"],
    )
    result["history_id"] = saved
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    return jsonify(result)


@app.get("/api/history")
def prediction_history():
    limit = request.args.get("limit", "12")
    try:
        limit_number = max(1, min(50, int(limit)))
    except ValueError:
        limit_number = 12
    return jsonify(history.latest(limit_number))


@app.delete("/api/history")
def clear_history():
    history.clear()
    return jsonify({"status": "cleared"})


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "sentiment_model": sentiment_model.status(),
            "digit_model": digit_model.status(),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
