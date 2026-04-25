# 🧠 NLP Sentiment Analysis — Production ML Pipeline

A production-ready **3-class sentiment analysis** system using a **Bidirectional LSTM** neural network.  
Classifies text as **Negative 😠 / Neutral 😐 / Positive 😊** with a FastAPI backend and Docker deployment.

---

## 📁 Project Structure

```
nlp-sentiment-project/
├── data/                          # Dataset (auto-generated)
│   └── reviews.csv
├── docs/                          # Training artefacts
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── metrics.json
├── notebooks/
│   └── exploration.ipynb
├── scripts/
│   ├── generate_data.py           # Synthetic data generator
│   └── demo.py                    # CLI demo
├── src/
│   ├── data_processing/
│   │   └── preprocessor.py        # Text cleaning + tokenisation
│   ├── models/
│   │   └── lstm_model.py          # Bidirectional LSTM architecture
│   ├── training/
│   │   └── train.py               # Full training pipeline
│   ├── inference/
│   │   └── predictor.py           # SentimentPredictor class
│   ├── api/
│   │   └── main.py                # FastAPI application
│   └── monitoring/
│       └── tracker.py             # In-process metrics tracker
├── tests/
│   └── test_pipeline.py           # pytest test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## ⚡ Quick Start (Local)

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Generate dataset (auto-done by training script)

```bash
python scripts/generate_data.py
```

### 3 — Train the model

```bash
python src/training/train.py
```

Training creates:
- `src/models/sentiment_model.h5` — trained model
- `src/data_processing/tokenizer.pkl` — fitted tokeniser
- `docs/training_curves.png` — accuracy/loss plots
- `docs/confusion_matrix.png` — evaluation heatmap
- `docs/metrics.json` — test metrics

### 4 — Run the CLI demo

```bash
python scripts/demo.py
```

### 5 — Start the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 6 — Run tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

```bash
# Build & start
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop
docker-compose down
```

---

## 🔌 API Reference

| Method | Endpoint         | Description                     |
|--------|------------------|---------------------------------|
| GET    | `/`              | Root / welcome                  |
| GET    | `/health`        | System health check             |
| GET    | `/metrics`       | Performance metrics snapshot    |
| POST   | `/predict`       | Single text prediction          |
| POST   | `/batch_predict` | Batch predictions (max 100)     |

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

**Response:**
```json
{
  "text": "This product is absolutely amazing!",
  "label": "Positive",
  "label_id": 2,
  "confidence": 0.9732,
  "emoji": "😊",
  "scores": { "Negative": 0.012, "Neutral": 0.0148, "Positive": 0.9732 },
  "latency_ms": 42.5
}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible quality", "It is okay"]}'
```

---

## 🧠 Model Architecture

```
Input (text)
    ↓
Embedding (vocab_size × 64)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
Dropout (0.4)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dropout (0.3)
    ↓
Dense (32, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (3, Softmax)  →  [Negative, Neutral, Positive]
```

| Hyperparameter    | Value         |
|-------------------|---------------|
| Max vocabulary    | 10,000 tokens |
| Max sequence len  | 100 tokens    |
| Embedding dim     | 64            |
| Optimiser         | Adam (lr=1e-3)|
| Loss              | Sparse Categorical Cross-entropy |
| Early stopping    | patience=4    |
| LR scheduler      | ReduceLROnPlateau |

---

## 📊 Expected Results

| Metric             | Value     |
|--------------------|-----------|
| Test Accuracy      | ~88–93%   |
| Weighted F1        | ~0.87–0.92|
| Avg API latency    | < 200 ms  |
| Batch throughput   | ~100 texts/s |

---

## 🔍 Ethical Considerations

- **Bias**: Model trained on synthetic data; retrain on real domain data before production use.
- **Edge cases**: Short/ambiguous texts return lower confidence — surface the score to users.
- **Privacy**: Never log raw user text in production; anonymise before monitoring.
- **Drift**: Monitor label distribution over time; retrain when drift is detected.

---

## 📈 Scalability

- **Horizontal**: Docker Swarm / Kubernetes ready (stateless API).
- **Vertical**: Increase `--workers` in uvicorn CMD.
- **Caching**: Add Redis layer for repeated queries.
- **Async batching**: FastAPI supports async endpoints for higher throughput.

---

## 🛠️ Tech Stack

| Component      | Technology           |
|----------------|----------------------|
| Deep learning  | TensorFlow / Keras   |
| API server     | FastAPI + Uvicorn    |
| Containerisation | Docker + Compose  |
| Testing        | pytest               |
| Logging        | Loguru               |
| Visualisation  | Matplotlib, Seaborn  |
