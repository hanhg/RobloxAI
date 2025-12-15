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
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": req.message},
        timeout=30
    )

    data = r.json()

    # Handle errors
    if isinstance(data, dict):
        if "error" in data:
            return {"reply": f"Error: {data['error']}"}
        if "generated_text" in data:
            return {"reply": data["generated_text"]}

    # Handle list response
    if isinstance(data, list) and len(data) > 0:
        if "generated_text" in data[0]:
            return {"reply": data[0]["generated_text"]}

    return {"reply": "No response from model."}
