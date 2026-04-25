"""
src/training/train.py
End-to-end training pipeline.
Run: python src/training/train.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from loguru import logger

from src.data_processing.preprocessor import TextPreprocessor, load_dataset
from src.models.lstm_model import build_model, get_callbacks, save_model


LABEL_NAMES = ["Negative", "Neutral", "Positive"]
MODEL_PATH  = "src/models/sentiment_model.h5"
PREP_PATH   = "src/data_processing/tokenizer.pkl"
REPORT_DIR  = "docs"


def plot_history(history, out_dir: str = REPORT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[metric[0]], label="Train")
        ax.plot(history.history[metric[1]], label="Val", linestyle="--")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Training curves saved → {path}")


def plot_confusion(y_true, y_pred, out_dir: str = REPORT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    logger.info(f"Confusion matrix saved → {path}")


def main():
    logger.info("=== NLP Sentiment Analysis — Training Pipeline ===")

    # 1. Load data
    csv_path = "data/reviews.csv"
    if not os.path.exists(csv_path):
        logger.info("Dataset not found — generating synthetic data …")
        os.makedirs("data", exist_ok=True)
        from scripts.generate_data import generate_dataset
        generate_dataset(n=3000, path=csv_path)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(csv_path)

    # 2. Preprocess
    prep = TextPreprocessor()
    X_train_enc = prep.fit_transform(X_train)
    X_val_enc   = prep.transform(X_val)
    X_test_enc  = prep.transform(X_test)
    prep.save(PREP_PATH)

    # 3. Build model
    model = build_model(vocab_size=prep.vocab_size)
    model.summary()

    # 4. Train
    history = model.fit(
        X_train_enc, y_train,
        validation_data=(X_val_enc, y_val),
        epochs=20,
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # 5. Evaluate
    loss, acc = model.evaluate(X_test_enc, y_test, verbose=0)
    logger.info(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    y_pred = np.argmax(model.predict(X_test_enc, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES)
    print("\n" + report)

    # 6. Save artefacts
    save_model(model, MODEL_PATH)
    plot_history(history)
    plot_confusion(y_test, y_pred)

    metrics = {
        "test_accuracy": round(float(acc), 4),
        "test_loss":     round(float(loss), 4),
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success("Training complete ✅")
    return metrics


if __name__ == "__main__":
    main()
