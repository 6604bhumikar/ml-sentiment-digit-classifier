# Deployment

This is now a Flask ML web application, not a Streamlit app.

## Render Settings

```text
Build command: pip install -r requirements.txt
Start command: gunicorn app:app
```

The included `render.yaml` can also be used for blueprint deployment.

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:5000
```
