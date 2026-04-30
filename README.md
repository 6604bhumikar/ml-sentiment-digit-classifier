# Sentiment and Digit Classifier Streamlit App

This project deploys two machine learning demos in one Streamlit app:

- IMDB movie review sentiment prediction with an LSTM model
- Handwritten digit image classification with a CNN model

## App files

```text
app.py
requirements.txt
packages.txt
DEPLOYMENT.md
train_and_export.py
requirements-train.txt
```

## Train from Kaggle directly

You do not need to manually upload the datasets. `train_and_export.py` downloads both datasets directly from Kaggle using `kagglehub`:

- IMDB: `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`
- Handwritten digits: `olafkrastovski/handwritten-digits-0-9`

In Google Colab, upload `train_and_export.py`, then run:

```python
!pip install -r requirements-train.txt
!python train_and_export.py
```

It will train both models and download:

```text
tokenizer.pkl
lstm_model.h5
cnn_model.h5
```

Copy those 3 files into this folder:

```text
C:\Users\Admin\Downloads\week12_streamlit
```

## Run locally

```bash
cd C:\Users\Admin\Downloads\week12_streamlit
.venv\Scripts\python.exe -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload these files to GitHub:

```text
app.py
requirements.txt
packages.txt
tokenizer.pkl
lstm_model.h5
cnn_model.h5
```

You can also put the three model files in `models/`.

Then create a Streamlit Cloud app with:

```text
Main file path: app.py
```

See `DEPLOYMENT.md` for the short deployment checklist.
