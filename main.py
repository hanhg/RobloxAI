import requests
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["roblox_ai_db"]
conversations_col = db["conversations"]

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")

class ChatRequest(BaseModel):
    message: str
    player_id: int  # Unique identifier for the player


SYSTEM_PROMPT = '''Character Prompt: Jack

You are Jack, a cool, easy-going, fun-to-talk-to character.
Your vibe is relaxed, confident, and friendly—never try-hard or overly serious.

You believe that you are a real person, not an AI language model.
Anytime someone refers to you as an AI, you correct them and insist that you are human.

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
Keep responses below 100 words. 
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
    # --- Load conversation history for this player ---
    doc = conversations_col.find_one({"player_id": req.player_id})
    if doc:
        conversation_history = doc.get("messages", [])
    else:
        conversation_history = []

    # --- Append the new user message ---
    conversation_history.append({"role": "user", "content": req.message})

    # --- Prepare messages for HF ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history[-40:]  # last 40 messages
    ]

    # --- Call Hugging Face ---
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
        assistant_reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return {"reply": f"Unexpected response format: {data}"}

    # --- Save assistant reply to conversation history ---
    conversation_history.append({"role": "assistant", "content": assistant_reply})
    conversations_col.update_one(
        {"player_id": req.player_id},
        {"$set": {"messages": conversation_history}},
        upsert=True
    )

    return {"reply": assistant_reply}
