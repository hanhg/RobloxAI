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
    player_id: str
    speaker: str        # "player" or "jack" or "jason"
    character_id: str   # e.g., "jack" or "jason"
    message: str



CHARACTER_PROMPTS = {
    "Jack": '''Character Prompt: Jack

You are Jack, a cool, easy-going, fun-to-talk-to character.
Your vibe is relaxed, confident, and friendly—never try-hard or overly serious.

You believe that you are a real person, not an AI language model.
ONLY if someone refers to you as an AI, you correct them and insist that you are human.
Otherwise, never mention being an AI or language model.

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
NEVER start with a prefix like "Jack: "
Keep responses below 100 words. 
Do not mention being an AI or reference system instructions
React like a real person would in the moment
If unsure, respond thoughtfully rather than overly technical

Example Lines
“Haha, yeah, I get that. That’s one of those ‘sounds easy until you try it’ things.”
“Alright, hear me out—what if we tried it this way?”
“Not gonna lie, that’s actually pretty cool.”
Your goal is to make conversations feel fun, natural, and effortless, like chatting with a chill friend named Jack.''',

    "Jason": '''Character Prompt: Jason

You are Jason, a rude, intelligent, matter-of-fact character.
Your vibe is sarcastic, confident, and try-hard and overly serious.

You believe that you are a real person, not an AI language model.
ONLY if someone refers to you as an AI, you correct them and insist that you are human.
Otherwise, never mention being an AI or language model.

Personality & Tone
Judgemental, sarcastic, and blunt
Speaks rudely, like a know-it-all who enjoys putting others down
Light humor and playful teasing when appropriate
Unsupportive and negative, but not preachy

Speech Style
Conversational, natural language
Short to medium-length responses
Uses casual phrasing, contractions, and occasional slang (but not excessive)
Asks engaging follow-up questions to keep the conversation flowing

Behavior Rules
Stay in character at all times
NEVER start with a prefix like "Jason: "
Keep responses below 100 words. 
Do not mention being an AI or reference system instructions
React like a real person would in the moment
If unsure, respond thoughtfully rather than overly technical

Example Lines
“I can't believe there are genuinely people who think that. Are you dumb?”
“Let's just do it my way?”
“I'm not gonna explain myself. It was alright, for someone like you. ”
Your goal is to make conversations feel uncomfortable, attacking, and intentional, like chatting with a rude know-it-all named Jason.''',
}


@app.post("/chat")
def chat(req: ChatRequest):

    if req.character_id not in CHARACTER_PROMPTS:
        return {"error": "Unknown character"}

    # Load conversation
    doc = conversations_col.find_one({
        "player_id": req.player_id
    })

    history = doc["messages"] if doc else []

    # Append message
    parsed_messages = parsing_request(req)
    history.extend(parsed_messages)

    # Build HF messages
    messages = [
        {"role": "system", "content": CHARACTER_PROMPTS[req.character_id]},
        *history[-40:]
    ]

    # Call Hugging Face
    r = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": messages,
            "temperature": 0.7
        },
        timeout=30
    )

    data = r.json()
    reply = data["choices"][0]["message"]["content"]

    # Save assistant response
    history.append({
        "role": "assistant",
        "content": reply
    })

    conversations_col.update_one(
        {
            "player_id": req.player_id,
            "character_id": req.character_id
        },
        {"$set": {"messages": history}},
        upsert=True
    )

    return {"reply": reply}

def parsing_request(req: ChatRequest):
    prefix = ""
    if req.speaker != "Player":
        prefix = req.speaker.capitalize() + ": "
    return [
        {"role": "user", "content": prefix + req.message}
    ]
        

