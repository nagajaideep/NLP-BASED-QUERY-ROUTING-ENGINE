# RouteIQ - AI Customer Support Router

FastAPI app that serves a static support-routing UI and a local DistilBERT
classification model for intelligent customer query routing.

## Overview

RouteIQ automatically classifies customer support queries into departments (account, general, orders, payments, returns) using a fine-tuned DistilBERT model, achieving 99.93% accuracy with sub-200ms inference latency.

## What is included

- `backend/main.py` - FastAPI app and `/predict` API.
- `frontend/index.html` - browser UI served from `/`.
- `backend/model/` - tokenizer, labels, and model weights.
- `Procfile` - deploy command for platforms that use Procfile-style web apps.
- `test_local_latency.py` - latency measurement script.
- `backend/test_model.py` - model loading smoke test.

## Local setup

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Model Performance

### Training & Evaluation Metrics

**Dataset Split:**
- Total examples: 26,872 customer support queries
- Training: 21,497 (80%)
- Validation: 2,687 (10%)
- Test: 2,688 (10%)
- Split method: Stratified by department label with `random_state=42`

**Test Set Results:**
- **Accuracy: 99.93%**
- **Precision: 0.9993**
- **Recall: 0.9993**
- **F1 Score: 0.9993**
- Misclassifications: 2 out of 2,688 test examples

**Model Architecture:**
- Base: DistilBERT (distilbert-base-uncased)
- Fine-tuned for sequence classification
- 5 output classes (departments)
- Model size: ~250MB (stored as `model.safetensors`)

### Latency Benchmarks

#### Local Inference (Recommended for Production)

Measured on localhost:8000 with FastAPI + DistilBERT on CPU.

**Configuration:**
- 100 total requests across 5 different query types
- 20 iterations per query type
- DistilBERT inference on CPU (no GPU)
- Includes full pipeline: tokenization + inference + response formatting

**Results:**
- **Average: 116.69 ms** ✓
- **Median: 111.66 ms**
- **P95: 163.48 ms**
- **Range: 83.19 - 229.51 ms**
- Success rate: 100%

**Measurement command:**
```powershell
python test_local_latency.py
```

#### Deployment Latency (Render Free Tier)

Measured against live deployment at https://nlp-based-query-routing-engine.onrender.com

**Results:**
- Average: ~11,000 ms
- Includes cold start delays and network latency
- Not representative of actual model inference speed

**Note:** The deployed latency is significantly higher due to Render's free tier cold starts and network overhead. Local inference provides accurate model performance metrics.

## Testing

### Smoke test

```powershell
venv\Scripts\python.exe backend\test_model.py
```

The test loads the tokenizer and model from `backend/model/`.

### Latency test

```powershell
# Start the server first
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# In another terminal, run the latency test
venv\Scripts\python.exe test_local_latency.py
```

Measures end-to-end request processing time over 100 requests.

## API

- `GET /` - serves the UI.
- `GET /health` - confirms the app is running and returns labels.
- `POST /predict` - classifies a customer query.

Example request:

```json
{
  "text": "My payment failed but the amount was deducted."
}
```

## Deployment notes

**Current deployment:** https://nlp-based-query-routing-engine.onrender.com

The app is configured for Python 3.11 and starts with:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

`backend/model/model.safetensors` is larger than GitHub's normal file limit, so
it is tracked through Git LFS via `.gitattributes`. Install and enable Git LFS
before pushing this app to GitHub:

```bash
git lfs install
git lfs track "backend/model/model.safetensors"
```

**Performance Note:** For production use cases requiring low latency, local or dedicated server deployment is recommended over free-tier hosting platforms due to cold start delays.

## Repository

- **GitHub**: https://github.com/nagajaideep/NLP-BASED-QUERY-ROUTING-ENGINE
- **Latest commit**: `2c47433f739ba662d9c10bb7fe4a1adf562dae7e`
