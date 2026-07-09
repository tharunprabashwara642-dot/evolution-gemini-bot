import os
import logging
from typing import Any, Dict, Optional
import httpx
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Environment variables
GROQ_API_KEY = os.getenv("GEMINI_API_KEY") # ඔයාගේ Groq API key එක මෙතනටම පාවිච්චි කරන්න
MODEL = os.getenv("GEMINI_MODEL", "llama3-70b-8192") # Groq model name එක මෙතන දාන්න
PORT = int(os.getenv("PORT", "8000"))

# Groq URL එක
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    logging.warning("Missing API key")

async def call_groq(prompt: str):
    """
    Groq AI වෙත ඉල්ලීම් යවන ශන් එක.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(GROQ_BASE_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

@app.post("/webhook")
async def webhook(request: Request):
    # මෙතනදී ඔයාගේ පණිවිඩය ලබාගන්න
    data = await request.json()
    prompt = data.get("message", "")
    
    try:
        reply_text = await call_groq(prompt)
        return {"reply": reply_text}
    except Exception as e:
        logging.error(f"Groq call failed: {e}")
        raise HTTPException(status_code=500, detail="Error calling Groq")
