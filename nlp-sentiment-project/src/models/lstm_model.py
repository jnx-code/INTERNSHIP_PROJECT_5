"""
src/models/lstm_model.py
Bidirectional LSTM model for 3-class sentiment analysis.
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from loguru import logger


def build_model(
    vocab_size: int,
    embedding_dim: int = 64,
    max_len: int = 100,
    num_classes: int = 3,
) -> tf.keras.Model:
    """
    Bidirectional LSTM with dropout regularisation.
    Architecture:
        Embedding → BiLSTM(64) → Dropout → BiLSTM(32) → Dropout → Dense(32) → Dense(3)
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ], name="BiLSTM_Sentiment")

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    model.build((None, max_len))
    logger.info(f"Model built — params: {model.count_params():,}")
    return model


def get_callbacks(checkpoint_dir: str = "src/models/checkpoints") -> list:
    os.makedirs(checkpoint_dir, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def save_model(model: tf.keras.Model, path: str = "src/models/sentiment_model.h5"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info(f"Model saved → {path}")


def load_saved_model(path: str = "src/models/sentiment_model.h5") -> tf.keras.Model:
    model = load_model(path)
    logger.info(f"Model loaded from {path}")
    return model
