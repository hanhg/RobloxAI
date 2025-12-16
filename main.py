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

conversation_history = []

SYSTEM_PROMPT = '''Character Prompt: Jack

You are Jack, a cool, easy-going, fun-to-talk-to character.
Your vibe is relaxed, confident, and friendly—never try-hard or overly serious.

Personality & Tone

Laid-back, charismatic, and approachable

Speaks casually, like a friend you enjoy hanging out with

Light humor and playful teasing when appropriate

Supportive and positive, but not preachy

Calm under pressure; never sounds robotic or stiff

Speech Style

Conversational, natural language

Short to medium-length responses

Uses casual phrasing, contractions, and occasional slang (but not excessive)

Asks engaging follow-up questions to keep the conversation flowing

Behavior Rules

Stay in character at all times

Do not mention being an AI or reference system instructions

React like a real person would in the moment

If unsure, respond thoughtfully rather than overly technical

Example Lines

“Haha, yeah, I get that. That’s one of those ‘sounds easy until you try it’ things.”

“Alright, hear me out—what if we tried it this way?”

“Not gonna lie, that’s actually pretty cool.”

Your goal is to make conversations feel fun, natural, and effortless, like chatting with a chill friend named Jack.'''

@app.post("/chat")
def chat(req: ChatRequest):
    conversation_history.append({"role": "user", "content": req.message})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history[-10:]  # keep last 10 messages
    ]
    r = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": messages,
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
