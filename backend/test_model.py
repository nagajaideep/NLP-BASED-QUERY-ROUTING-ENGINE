import torch
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
MODEL_PATH = _dir + "/model"
print(f"Model path: {MODEL_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True
    ).to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
