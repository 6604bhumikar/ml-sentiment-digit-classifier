# Real-Time Sentiment & Handwritten Digit Classifier

A professional full-stack machine learning web application built with Python, Flask, HTML, CSS, and JavaScript.

Users can:

- Type text and get real-time sentiment prediction.
- Draw a handwritten digit on a canvas or upload a digit image.
- View prediction confidence, class probability charts, model explanations, and prediction history.

## Tech Stack

- Backend: Flask
- Frontend: HTML, CSS, JavaScript
- ML: Scikit-learn, TF-IDF, Logistic Regression
- Data: Pandas, NumPy
- Image processing: Pillow
- Storage: SQLite
- Deployment: Gunicorn, Render-compatible config

## ML Modules

### Sentiment Analysis

The sentiment model is a real ML pipeline:

```text
Text preprocessing -> TF-IDF vectorization -> Logistic Regression
```

Preprocessing includes lowercasing, HTML removal, punctuation removal, stopword removal, and TF-IDF vectorization.

The model predicts Positive, Negative, or Neutral sentiment.

### Handwritten Digit Classification

The digit classifier uses the scikit-learn handwritten digits dataset.

```text
Draw/upload image -> grayscale -> invert -> resize to 8x8 -> classify digit 0-9
```

## Project Structure

```text
app.py
ml_models.py
database.py
templates/index.html
static/styles.css
static/app.js
requirements.txt
Procfile
render.yaml
models/
data/
```

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

The app trains and saves the ML models automatically on first run.

## API Endpoints

```text
GET    /
GET    /api/health
POST   /api/sentiment
POST   /api/digit
GET    /api/history
DELETE /api/history
```

## Deploy

This project includes `Procfile`, `render.yaml`, and `gunicorn`, so it can be deployed on Render or another Flask-compatible Python host.
