"""Download Kaggle datasets, train, and export Week 12 model artifacts.

Run this in Google Colab or locally from the Week 12 project folder.
It creates:
- tokenizer.pkl
- lstm_model.h5
- cnn_model.h5
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import kagglehub
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

IMDB_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
DIGIT_DATASET = "olafkrastovski/handwritten-digits-0-9"
MAX_WORDS = 5000
MAX_LEN = 100
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_file(root_path: str, filename: str) -> Path:
    for root, _, files in os.walk(root_path):
        if filename in files:
            return Path(root) / filename
    raise FileNotFoundError(f"Could not find {filename} inside {root_path}")


def download_imdb_csv() -> Path:
    print("Downloading IMDB dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download(IMDB_DATASET)
    csv_path = find_file(dataset_path, "IMDB Dataset.csv")
    print(f"IMDB CSV: {csv_path}")
    return csv_path


def train_lstm() -> None:
    csv_path = download_imdb_csv()
    data = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", encoding="latin1")
    data["Sentiment"] = data["sentiment"].map({"positive": 1, "negative": 0})
    data = data.dropna(subset=["review", "Sentiment"])
    data["cleaned_review"] = data["review"].apply(clean_text)

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(data["cleaned_review"])
    x = tokenizer.texts_to_sequences(data["cleaned_review"])
    y = data["Sentiment"].astype(int)

    with open("tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")

    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        LSTM(64),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, validation_split=0.1)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"LSTM test loss: {loss:.4f}")
    print(f"LSTM test accuracy: {accuracy:.4f}")
    model.save("lstm_model.h5")


def find_digit_base_path(download_path: str) -> str:
    for root, dirs, _ in os.walk(download_path):
        digit_dirs = {str(i) for i in range(10)}
        if digit_dirs.issubset(set(dirs)):
            return root
    raise FileNotFoundError("Could not find digit class folders 0-9 after Kaggle download.")


def train_cnn() -> None:
    print("Downloading handwritten digit dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download(DIGIT_DATASET)
    base_path = find_digit_base_path(dataset_path)
    print(f"Digit image folder: {base_path}")

    rows = []
    for label in sorted(os.listdir(base_path)):
        class_path = os.path.join(base_path, label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                rows.append([os.path.join(class_path, filename), label])

    df = pd.DataFrame(rows, columns=["image_path", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        x_col="image_path",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    validation_generator = eval_datagen.flow_from_dataframe(
        dataframe=val_data,
        x_col="image_path",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_generator = eval_datagen.flow_from_dataframe(
        dataframe=test_data,
        x_col="image_path",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    cnn_model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    cnn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    cnn_model.fit(train_generator, epochs=30, validation_data=validation_generator, verbose=1)
    loss, accuracy = cnn_model.evaluate(test_generator, verbose=1)
    print(f"CNN test loss: {loss:.4f}")
    print(f"CNN test accuracy: {accuracy:.4f}")
    cnn_model.save("cnn_model.h5")


def download_from_colab() -> None:
    try:
        from google.colab import files
    except ImportError:
        print("Not running in Colab. Files are saved in the current folder.")
        return

    files.download("tokenizer.pkl")
    files.download("lstm_model.h5")
    files.download("cnn_model.h5")


if __name__ == "__main__":
    train_lstm()
    train_cnn()
    download_from_colab()
