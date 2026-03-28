
# NLP-Based Query Routing Engine

An AI-powered customer support query router built with FastAPI, DistilBERT, and a single-page frontend.

The app classifies incoming customer messages and routes them to the most relevant support department, returning:
- predicted department  
- confidence score  
- full score distribution across departments  
- urgency flag  
- matched keywords from the query  

---

## Features

- DistilBERT sequence classification model loaded from local model files  
- FastAPI backend with REST endpoints  
- Browser-based frontend for live routing and visualization  
- Real-time confidence bars and score breakdown  
- Query history and department routing counters  
- CORS enabled for easy frontend-backend communication  

---

## Department Classes

Current model labels:
- account  
- general  
- orders  
- payments  
- returns  

---

## Project Structure

```

.
├── main.py              # FastAPI app, model loading, prediction logic
├── index.html           # Frontend UI and client-side logic
├── model/               # Tokenizer, model weights, labels.json
├── requirements.txt     # Dependencies
├── Procfile             # Deployment command

````

---

## Tech Stack

- Python  
- FastAPI  
- Uvicorn  
- PyTorch  
- Hugging Face Transformers (DistilBERT)  
- HTML, CSS, JavaScript  

---

## API Endpoints

### GET /
Serves the frontend page  

---

### POST /predict

Classifies a query into a department  

**Request**
```json
{
  "text": "my order has not arrived yet"
}
````

**Response**

```json
{
  "department": "orders",
  "confidence": 0.9421,
  "all_scores": {
    "account": 0.0123,
    "general": 0.0211,
    "orders": 0.9421,
    "payments": 0.0102,
    "returns": 0.0143
  },
  "is_urgent": false,
  "matched_keywords": ["order", "arrived"]
}
```

---

### GET /health

Returns API health status and loaded labels

---

## Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/nagajaideep/NLP-BASED-QUERY-ROUTING-ENGINE.git
cd NLP-BASED-QUERY-ROUTING-ENGINE
```

### 2. Create Virtual Environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Open App

```
http://127.0.0.1:8000
```

---

## Deployment

**Procfile**

```bash
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Frontend behavior:

* Uses `http://127.0.0.1:8000` for local development
* Uses production API URL when deployed

---

## Notes

* Model is loaded from the `model/` directory
* Ensure all model files and `labels.json` are present
* Use Git LFS if model files exceed size limits

---

