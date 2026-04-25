"""
src/data_processing/preprocessor.py
Text preprocessing pipeline for NLP sentiment analysis.
"""
import re
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from loguru import logger


MAX_WORDS = 10_000
MAX_LEN   = 100
OOV_TOKEN = "<OOV>"


def clean_text(text: str) -> str:
    """Lowercase, strip HTML, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # keep alphanumerics
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextPreprocessor:
    def __init__(self, max_words: int = MAX_WORDS, max_len: int = MAX_LEN):
        self.max_words = max_words
        self.max_len   = max_len
        self.tokenizer: Tokenizer | None = None

    # ------------------------------------------------------------------ #
    def fit(self, texts: list[str]):
        clean = [clean_text(t) for t in texts]
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token=OOV_TOKEN)
        self.tokenizer.fit_on_texts(clean)
        logger.info(f"Tokenizer fitted — vocab size: {len(self.tokenizer.word_index):,}")
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError("Call fit() before transform().")
        clean = [clean_text(t) for t in texts]
        seqs  = self.tokenizer.texts_to_sequences(clean)
        return pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    # ------------------------------------------------------------------ #
    def save(self, path: str = "src/data_processing/tokenizer.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str = "src/data_processing/tokenizer.pkl") -> "TextPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Preprocessor loaded from {path}")
        return obj

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            return 0
        return min(len(self.tokenizer.word_index) + 1, self.max_words)


# ------------------------------------------------------------------ #
def load_dataset(csv_path: str = "data/reviews.csv"):
    """Load CSV with columns [text, label] and return train/val/test splits."""
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset loaded: {len(df):,} rows | labels: {sorted(df['label'].unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].values,
        test_size=0.15, random_state=42, stratify=df["label"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15, random_state=42, stratify=y_train
    )

    logger.info(f"Split — train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
