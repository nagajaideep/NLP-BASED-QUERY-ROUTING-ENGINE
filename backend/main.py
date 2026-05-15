import json
import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import re

app = FastAPI(title="RouteIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Load model ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "model").replace("\\", "/")
FRONTEND_INDEX = os.path.join(PROJECT_ROOT, "frontend", "index.html")
print(f"Model path: {MODEL_PATH}")

device = torch.device("cpu")
print(f"Forcing model on {device} for local run...")

print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
print("Loading model (this may take a moment)...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH, local_files_only=True
).to(device)

model.eval()

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "labels.json")
) as f:
    LABELS = json.load(f)

print(f"Model loaded OK. Labels: {LABELS}")

ANGRY_WORDS = {
    "angry",
    "furious",
    "terrible",
    "awful",
    "ridiculous",
    "worst",
    "horrible",
    "unacceptable",
    "disgusting",
    "fraud",
    "scam",
    "useless",
    "pathetic",
    "outrageous",
    "incompetent",
}


# ── Serve index.html at root ──
@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_INDEX)


# ── Schemas ──
class Query(BaseModel):
    text: str


class PredictResponse(BaseModel):
    department: str
    confidence: float
    all_scores: dict[str, float]
    is_urgent: bool
    matched_keywords: list[str]


# ── Predict endpoint ──
@app.post("/predict", response_model=PredictResponse)
def predict(query: Query):
    text = query.text.strip()[:512]
    if not text:
        raise HTTPException(status_code=400, detail="Query text is required.")

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128, padding="max_length"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]

    scores = {LABELS[str(i)]: round(probs[i].item(), 4) for i in range(len(LABELS))}

    best_idx = probs.argmax().item()
    best_dept = LABELS[str(best_idx)]
    best_conf = round(probs[best_idx].item(), 4)

    stopwords = {
        "that",
        "this",
        "with",
        "have",
        "from",
        "been",
        "they",
        "them",
        "your",
        "just",
        "will",
        "when",
        "what",
        "about",
    }
    tokens = re.findall(r"[a-zA-Z']+", text.lower())

    matched = []
    seen = set()
    for t in tokens:
        if len(t) > 3 and t not in stopwords and t not in seen:
            seen.add(t)
            matched.append(t)
    matched = matched[:6]

    is_urgent = any(t in ANGRY_WORDS for t in tokens)

    return PredictResponse(
        department=best_dept,
        confidence=best_conf,
        all_scores=scores,
        is_urgent=is_urgent,
        matched_keywords=matched,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "RouteIQ API",
        "model_path": MODEL_PATH,
        "labels": LABELS,
    }
