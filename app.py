from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import os
import requests

app = FastAPI()

# -------------------------
# Hugging Face model setup
# -------------------------
hf_model_id = "Manasa10/Mood-Predictor"  # <- replace with your actual model repo if different

# Paths
label_encoder_url = f"https://huggingface.co/{hf_model_id}/resolve/main/label_encoder.pkl"
label_encoder_path = "label_encoder.pkl"

# Download label_encoder.pkl if not present
if not os.path.exists(label_encoder_path):
    print("Downloading label encoder from Hugging Face...")
    response = requests.get(label_encoder_url)
    with open(label_encoder_path, "wb") as f:
        f.write(response.content)

# Load model and tokenizer from Hugging Face Hub
model = BertForSequenceClassification.from_pretrained(hf_model_id)
tokenizer = BertTokenizer.from_pretrained(hf_model_id)
le = joblib.load(label_encoder_path)

model.eval()

# -------------------------
# Request body model
# -------------------------
class TextInput(BaseModel):
    text: str

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/moodpredictor")
def predict_mood(data: TextInput):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    mood = le.inverse_transform([prediction])[0]
    return {"mood": mood}
