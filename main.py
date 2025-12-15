import requests
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    r = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": req.message},
        timeout=30
    )

    data = r.json()

    # Hugging Face sometimes returns errors as dicts
    if isinstance(data, dict) and "error" in data:
        return {"error": data["error"]}

    return {"reply": data[0]["generated_text"]}
