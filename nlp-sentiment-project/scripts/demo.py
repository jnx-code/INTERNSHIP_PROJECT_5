"""
scripts/demo.py
Quick CLI demo — runs predictions on sample texts.
Run AFTER training: python scripts/demo.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.predictor import SentimentPredictor

SAMPLES = [
    "This product is absolutely amazing! I love it so much.",
    "Worst purchase of my life. Completely fell apart after two days.",
    "It's okay, does what it says. Nothing special though.",
    "Five stars! Highly recommend to everyone.",
    "Terrible quality, do not buy this.",
    "Average product — decent for the price I suppose.",
    "Exceeded all my expectations. Outstanding!",
    "I'm very disappointed. Broke within a week.",
]


def main():
    print("\n🤖  Sentiment Analysis Demo")
    print("=" * 55)
    predictor = SentimentPredictor()
    print()
    for text in SAMPLES:
        r = predictor.predict_one(text)
        bar = "█" * int(r["confidence"] * 20)
        print(f"{r['emoji']}  [{r['label']:8s}]  {r['confidence']*100:5.1f}%  {bar}")
        print(f"   \"{text[:70]}\"")
        print()


if __name__ == "__main__":
    main()
