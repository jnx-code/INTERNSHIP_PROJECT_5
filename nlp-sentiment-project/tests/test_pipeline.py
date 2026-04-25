"""
tests/test_pipeline.py
Unit tests for preprocessing and inference logic.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.data_processing.preprocessor import TextPreprocessor, clean_text


# ------------------------------------------------------------------ #
# clean_text
# ------------------------------------------------------------------ #
def test_clean_text_lowercase():
    assert clean_text("Hello World") == "hello world"

def test_clean_text_removes_html():
    assert "<b>" not in clean_text("<b>bold</b>")

def test_clean_text_removes_punctuation():
    result = clean_text("Great!!!  😊")
    assert "!" not in result

def test_clean_text_strips_whitespace():
    assert clean_text("  hello  world  ") == "hello world"


# ------------------------------------------------------------------ #
# TextPreprocessor
# ------------------------------------------------------------------ #
SAMPLE_TEXTS = [
    "This is a great product!",
    "Terrible experience, very bad.",
    "It's okay, average at best.",
    "Highly recommended to everyone.",
    "Would not buy again.",
]


def test_preprocessor_fit_transform_shape():
    prep = TextPreprocessor(max_words=100, max_len=10)
    result = prep.fit_transform(SAMPLE_TEXTS)
    assert result.shape == (5, 10)

def test_preprocessor_transform_new_texts():
    prep = TextPreprocessor(max_words=100, max_len=10)
    prep.fit(SAMPLE_TEXTS)
    new_texts = ["New review here", "Another one"]
    result = prep.transform(new_texts)
    assert result.shape == (2, 10)

def test_preprocessor_oov_handling():
    prep = TextPreprocessor(max_words=50, max_len=5)
    prep.fit(["hello world"])
    result = prep.transform(["completely unknown words xyz"])
    assert result.shape == (1, 5)

def test_preprocessor_vocab_size():
    prep = TextPreprocessor(max_words=1000, max_len=10)
    prep.fit(SAMPLE_TEXTS)
    assert prep.vocab_size > 0
    assert prep.vocab_size <= 1001

def test_preprocessor_save_load(tmp_path):
    prep = TextPreprocessor(max_words=100, max_len=10)
    prep.fit(SAMPLE_TEXTS)
    path = str(tmp_path / "tokenizer.pkl")
    prep.save(path)
    loaded = TextPreprocessor.load(path)
    orig   = prep.transform(SAMPLE_TEXTS)
    reloaded = loaded.transform(SAMPLE_TEXTS)
    np.testing.assert_array_equal(orig, reloaded)

def test_preprocessor_requires_fit_before_transform():
    prep = TextPreprocessor()
    with pytest.raises(RuntimeError):
        prep.transform(["some text"])


# ------------------------------------------------------------------ #
# Monitoring
# ------------------------------------------------------------------ #
from src.monitoring.tracker import MetricsTracker

def test_tracker_records_requests():
    t = MetricsTracker()
    t.record("Positive", 120.5)
    t.record("Negative", 80.0)
    snap = t.snapshot()
    assert snap["total_requests"] == 2
    assert snap["error_count"] == 0

def test_tracker_error_rate():
    t = MetricsTracker()
    t.record("Positive", 100.0)
    t.record("error", 0, error=True)
    snap = t.snapshot()
    assert snap["error_rate_pct"] == 50.0

def test_tracker_avg_latency():
    t = MetricsTracker()
    t.record("Positive", 100.0)
    t.record("Neutral", 200.0)
    snap = t.snapshot()
    assert snap["avg_latency_ms"] == 150.0
