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
        "https://router.huggingface.co/models/mistralai/Mistral-7B-Instruct",
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"inputs": req.message},
        timeout=30
    )

    # 🔍 If HF returns non-JSON, don't crash
    try:
        data = r.json()
    except ValueError:
        return {
            "reply": f"Hugging Face returned non-JSON response (status {r.status_code})"
        }

    # Handle HF error responses
    if isinstance(data, dict) and "error" in data:
        return {"reply": f"HF Error: {data['error']}"}

    # Handle normal generation
    if isinstance(data, list) and len(data) > 0:
        if "generated_text" in data[0]:
            return {"reply": data[0]["generated_text"]}

    return {"reply": "No response from model."}
