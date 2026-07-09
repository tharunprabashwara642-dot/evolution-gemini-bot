import os
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EVOLUTION_SEND_APIKEY = os.getenv("EVOLUTION_SEND_APIKEY")
EVOLUTION_SERVER_URL = os.getenv("EVOLUTION_SERVER_URL")  # optional fallback
PORT = int(os.getenv("PORT", "8000"))

if not GEMINI_API_KEY or not EVOLUTION_SEND_APIKEY:
    logging.warning("Missing required env vars: GEMINI_API_KEY and/or EVOLUTION_SEND_APIKEY")

def build_gemini_url(model: str, api_key: str) -> str:
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    return f"{base}/{model}:generateContent?key={api_key}"

async def call_gemini(prompt: str, max_output_tokens: int = 512) -> str:
    """
    Calls Google Generative Language (Gemini) REST endpoint using API key as query param.
    Request body: {"contents": [{"parts": [{"text": prompt}]}]}
    Response text: candidates[0].content.parts[0].text
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = build_gemini_url(GEMINI_MODEL, GEMINI_API_KEY)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        # optional: control tokens/params if provider supports
        # "maxOutputTokens": max_output_tokens
    }
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        j = resp.json()

    # Parse according to spec: candidates[0].content.parts[0].text
    try:
        candidates = j.get("candidates")
        if not candidates or not isinstance(candidates, list):
            raise ValueError("No candidates in response")
        first = candidates[0]
        content = first.get("content", {})
        # content may be dict with "parts"
        parts = content.get("parts")
        if parts and isinstance(parts, list) and len(parts) > 0:
            text = parts[0].get("text")
            if isinstance(text, str):
                return text.strip()
    except Exception as e:
        logging.exception("Failed parsing Gemini response: %s", e)

    # fallback: stringify entire response truncated
    return str(j)[:2000]

def build_prompt(sender_name: Optional[str], incoming_text: str) -> str:
    system_instr = (
        "ඔබ මිතුරෙක් වගේ, හුරුබුහුටි හා මිත්‍රශීලී Sinhala (සිංහල) භාෂාවෙන් පමණක් පිළිතුරු දෙන AI ය. "
        "Userට එන ඕනෑම පණිවිඩයක් (English/Tamil/Singlish/සිංහල) සඳහාම Sinhala භාෂාවෙන්, සරල හා මිත්‍රශීලී ශෛලියෙන් උත්තර දෙන්න. "
        "අවශ්‍ය නම් පොදු උදාහරණ දෙන්න, නමුත් concise (ඉතා දිගු නොවන) ලෙස විය යුතුයි."
    )
    user_context = f"Sender: {sender_name}\nMessage: {incoming_text}"
    prompt = f"{system_instr}\n\n{user_context}\n\nReply in Sinhala (friendly) only:"
    return prompt

@app.post("/webhook")
async def webhook(request: Request):
    body: Dict[str, Any] = await request.json()
    logging.info("Incoming webhook event: %s", body.get("event"))

    instance = body.get("instance")
    server_url = body.get("server_url") or EVOLUTION_SERVER_URL
    data = body.get("data", {}) or {}

    key = data.get("key", {}) or {}
    remote_jid = key.get("remoteJid", "")
    number = remote_jid.split("@")[0] if "@" in remote_jid else remote_jid

    push_name = data.get("pushName")
    message = data.get("message", {}) or {}
    incoming_text = (
        message.get("conversation")
        or (message.get("extendedTextMessage") or {}).get("text")
        or (message.get("text") if isinstance(message.get("text"), str) else None)
        or ""
    )

    if not incoming_text:
        logging.info("No text found in incoming message. Ignoring.")
        return {"status": "ignored", "reason": "no_text"}

    prompt = build_prompt(push_name or "User", incoming_text)
    try:
        reply_text = await call_gemini(prompt)
    except Exception as e:
        logging.exception("Gemini call failed")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    if not server_url:
        logging.error("No server_url provided in payload and EVOLUTION_SERVER_URL not set.")
        raise HTTPException(status_code=400, detail="No server_url available to send reply")

    send_url = f"{server_url.rstrip('/')}/message/sendText/{instance}"
    send_payload = {"number": number, "text": reply_text}
    headers = {"apikey": EVOLUTION_SEND_APIKEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        send_resp = await client.post(send_url, json=send_payload, headers=headers)
        if send_resp.status_code >= 400:
            logging.error("Failed to send reply: %s %s", send_resp.status_code, send_resp.text)
            raise HTTPException(status_code=502, detail="Failed to send reply to Evolution Manager")

    logging.info("Reply sent to %s", number)
    return {"status": "ok"}
