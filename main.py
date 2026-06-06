import uuid
from contextlib import asynccontextmanager
from typing import Optional
import json

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config import settings
from core.llm import llm_client, detect_language
from core.memory import memory_manager
from core.tts import tts_engine
from core.stt import stt_engine


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await memory_manager.connect()
    yield
    await memory_manager.disconnect()
    await llm_client.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NOVA",
    description="Karl's personal AI assistant",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    language: str
    session_id: str
    conversation_id: int

class SpeakRequest(BaseModel):
    text: str
    language: str = "en"

class FactRequest(BaseModel):
    category: str
    key: str
    value: str
    source: str = "user"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health():
    ollama_ok = await llm_client.check_availability()
    return {
        "status": "ok",
        "nova": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "ollama": {
            "available": ollama_ok,
            "model": settings.OLLAMA_MODEL,
            "url": settings.OLLAMA_BASE_URL,
        },
        "tts": {
            "available": tts_engine.is_available(),
            "mode": tts_engine.mode(),
            "remote_url": settings.NOVA_SERVER_URL,
            "voices": tts_engine.voices_available(),
        },
        "stt": {
            "available": stt_engine.is_available(),
            "model": settings.WHISPER_MODEL,
        },
        "claude_api": bool(settings.ANTHROPIC_API_KEY),
    }


# ── Chat (blocking) ───────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    conv_id = await memory_manager.get_or_create_session(session_id)
    history = await memory_manager.get_conversation_history(conv_id)
    facts = await memory_manager.get_facts()

    lang = detect_language(req.message)
    await memory_manager.save_message(conv_id, "user", req.message, lang)

    response = await llm_client.chat(req.message, history, facts)
    resp_lang = detect_language(response)
    await memory_manager.save_message(conv_id, "assistant", response, resp_lang)

    return ChatResponse(
        response=response,
        language=resp_lang,
        session_id=session_id,
        conversation_id=conv_id,
    )


# ── Chat (streaming SSE) ──────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Server-Sent Events stream.

    Event types:
      { "type": "meta",  "session_id": "...", "conversation_id": 123 }
      { "type": "token", "content": "..." }
      { "type": "done",  "language": "fr" }
      { "type": "error", "message": "..." }
    """
    session_id = req.session_id or str(uuid.uuid4())
    conv_id = await memory_manager.get_or_create_session(session_id)
    history = await memory_manager.get_conversation_history(conv_id)
    facts = await memory_manager.get_facts()

    lang = detect_language(req.message)
    await memory_manager.save_message(conv_id, "user", req.message, lang)

    async def generate():
        full_response = []
        try:
            yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id, 'conversation_id': conv_id})}\n\n"

            async for token in llm_client.chat_stream(req.message, history, facts):
                full_response.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            response_text = "".join(full_response)
            resp_lang = detect_language(response_text)
            await memory_manager.save_message(conv_id, "assistant", response_text, resp_lang)

            yield f"data: {json.dumps({'type': 'done', 'language': resp_lang})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── TTS ───────────────────────────────────────────────────────────────────────

@app.post("/speak")
async def speak(req: SpeakRequest):
    if not tts_engine.is_available():
        raise HTTPException(
            503,
            "TTS unavailable: install Piper locally or set NOVA_SERVER_URL in .env.",
        )
    try:
        audio = await tts_engine.synthesize(req.text, req.language)
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Remote TTS server error: {e.response.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(502, f"Remote TTS server unreachable: {e}")
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=nova.wav"},
    )


# ── STT ───────────────────────────────────────────────────────────────────────

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not stt_engine.is_available():
        raise HTTPException(503, "openai-whisper not installed. Run: pip install openai-whisper")
    raw = await audio.read()
    ext = (audio.filename or "audio.wav").rsplit(".", 1)[-1]
    result = await stt_engine.transcribe(raw, ext)
    return {"text": result["text"], "language": result["language"]}


# ── Memory ────────────────────────────────────────────────────────────────────

@app.get("/memory/sessions")
async def get_sessions():
    sessions = await memory_manager.get_recent_sessions()
    return {
        "sessions": [
            {**s, "session_id": str(s["session_id"]),
             "created_at": s["created_at"].isoformat(),
             "updated_at": s["updated_at"].isoformat()}
            for s in sessions
        ]
    }


@app.get("/memory/history/{session_id}")
async def get_history(session_id: str, limit: int = Query(50, ge=1, le=200)):
    conv_id = await memory_manager.get_or_create_session(session_id)
    history = await memory_manager.get_conversation_history(conv_id, limit=limit)
    return {
        "session_id": session_id,
        "conversation_id": conv_id,
        "messages": [
            {**m, "created_at": m["created_at"].isoformat() if m.get("created_at") else None}
            for m in history
        ],
    }


@app.get("/memory/facts")
async def get_facts(category: Optional[str] = None):
    facts = await memory_manager.get_facts(category)
    return {
        "facts": [
            {**f,
             "created_at": f["created_at"].isoformat(),
             "updated_at": f["updated_at"].isoformat()}
            for f in facts
        ]
    }


@app.post("/memory/facts")
async def save_fact(fact: FactRequest):
    await memory_manager.upsert_fact(fact.category, fact.key, fact.value, fact.source)
    return {"status": "saved", "fact": fact.model_dump()}


@app.delete("/memory/facts/{category}/{key}")
async def delete_fact(category: str, key: str):
    await memory_manager.delete_fact(category, key)
    return {"status": "deleted", "category": category, "key": key}
