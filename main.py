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
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "user", "content": req.message}
            ],
            "temperature": 0.7
        },
        timeout=30
    )

    try:
        data = r.json()
    except ValueError:
        return {"reply": f"Non-JSON response: {r.text}"}

    if "error" in data:
        return {"reply": f"HF Error: {data['error']}"}

    try:
        return {
            "reply": data["choices"][0]["message"]["content"]
        }
    except (KeyError, IndexError):
        return {
            "reply": f"Unexpected response format: {data}"
        }
