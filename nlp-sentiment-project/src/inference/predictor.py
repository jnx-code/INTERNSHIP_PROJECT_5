"""
src/inference/predictor.py
Load trained model and run predictions.
"""
import numpy as np
from loguru import logger

from src.data_processing.preprocessor import TextPreprocessor
from src.models.lstm_model import load_saved_model

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJI_MAP  = {0: "😠", 1: "😐", 2: "😊"}


class SentimentPredictor:
    """Wrapper around the trained model for production inference."""

    def __init__(
        self,
        model_path:  str = "src/models/sentiment_model.h5",
        prep_path:   str = "src/data_processing/tokenizer.pkl",
    ):
        self.preprocessor = TextPreprocessor.load(prep_path)
        self.model        = load_saved_model(model_path)
        logger.info("SentimentPredictor ready")

    # ------------------------------------------------------------------ #
    def predict_one(self, text: str) -> dict:
        enc   = self.preprocessor.transform([text])
        probs = self.model.predict(enc, verbose=0)[0]
        label_id   = int(np.argmax(probs))
        confidence = float(probs[label_id])
        return {
            "text":       text,
            "label":      LABEL_MAP[label_id],
            "label_id":   label_id,
            "confidence": round(confidence, 4),
            "emoji":      EMOJI_MAP[label_id],
            "scores": {
                "Negative": round(float(probs[0]), 4),
                "Neutral":  round(float(probs[1]), 4),
                "Positive": round(float(probs[2]), 4),
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        enc   = self.preprocessor.transform(texts)
        probs = self.model.predict(enc, verbose=0)
        results = []
        for text, prob in zip(texts, probs):
            label_id   = int(np.argmax(prob))
            results.append({
                "text":       text,
                "label":      LABEL_MAP[label_id],
                "label_id":   label_id,
                "confidence": round(float(prob[label_id]), 4),
                "emoji":      EMOJI_MAP[label_id],
                "scores": {
                    "Negative": round(float(prob[0]), 4),
                    "Neutral":  round(float(prob[1]), 4),
                    "Positive": round(float(prob[2]), 4),
                },
            })
        return results
