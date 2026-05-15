# RouteIQ - AI Customer Support Router

FastAPI app that serves a static support-routing UI and a local DistilBERT
classification model.

## What is included

- `backend/main.py` - FastAPI app and `/predict` API.
- `frontend/index.html` - browser UI served from `/`.
- `backend/model/` - tokenizer, labels, and model weights.
- `Procfile` - deploy command for platforms that use Procfile-style web apps.

## Local setup

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Smoke test

```powershell
venv\Scripts\python.exe backend\test_model.py
```

The test loads the tokenizer and model from `backend/model/`.

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
