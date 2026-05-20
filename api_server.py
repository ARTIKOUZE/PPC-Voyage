"""Serveur HTTP FastAPI qui expose le pipeline NL → CP-SAT au frontend React.
Endpoints :"""

from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator import handle_turn, reset_session, get_session_state
from llm_client import LLM_BASE_URL, LLM_MODEL

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Travel Planner CP-SAT + LLM", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str
    solve_timeout: int = Field(default=10, ge=1, le=60)
    mode: str = Field(default="flexible", pattern="^(flexible|strict)$")
    transport_mode: str | None = Field(default=None, pattern="^(foot|bike|car)$")

class ResetRequest(BaseModel):
    session_id: str = Field(default="default")

@app.get("/health")
def health():
    return {
        "ok": True,
        "llm_endpoint": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
    }

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message vide")

    result = handle_turn(
        session_id=req.session_id,
        user_message=req.message,
        solve_timeout=req.solve_timeout,
        mode=req.mode,
        transport_mode=req.transport_mode,
    )
    return result

@app.get("/state")
def state(session_id: str = "default"):
    return {"constraints": get_session_state(session_id)}

@app.post("/reset")
def reset(req: ResetRequest):
    reset_session(req.session_id)
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=os.environ.get("API_HOST", "127.0.0.1"),
        port=int(os.environ.get("API_PORT", 8000)),
        reload=True,
    )
