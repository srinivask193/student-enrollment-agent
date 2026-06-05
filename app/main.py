"""
FastAPI Server v3.0 - Async + Streaming Support
Author: Srinivas Kanukolanu
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import asyncio
from app.agent import EnrollmentAgent
from app.observability import logger

app = FastAPI(
    title="Student Enrollment Assistant API v3.0",
    description="Agentic AI — LangGraph + CrewAI + RAG + Memory + Guardrails + Observability",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Session store
sessions: dict[str, EnrollmentAgent] = {}


# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    use_crew: Optional[bool] = False  # Use CrewAI multi-agent


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_response: str
    agent_type: str


# ─────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────

def get_or_create_session(session_id: Optional[str]) -> tuple[str, EnrollmentAgent]:
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = EnrollmentAgent(session_id=session_id)
        logger.info(f"NEW_SESSION | id={session_id[:8]}")
    return session_id, sessions[session_id]


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Student Enrollment Assistant v3.0",
        "status": "running",
        "features": [
            "LangGraph agent loop",
            "CrewAI multi-agent (4 agents)",
            "RAG with ChromaDB + HuggingFace",
            "Short-term + Long-term memory (SQLite)",
            "Guardrails + prompt injection protection",
            "Observability + logging",
            "Streaming responses",
            "Message trimming"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(sessions)
    }


@app.post("/session/new")
async def new_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = EnrollmentAgent(session_id=session_id)
    return {"session_id": session_id, "message": "Session created."}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Standard chat endpoint."""
    session_id, agent = get_or_create_session(request.session_id)

    if request.use_crew:
        response = agent.chat_with_crew(request.message)
        agent_type = "CrewAI Multi-Agent (4 agents)"
    else:
        response = agent.chat(request.message)
        agent_type = "LangGraph Single Agent"

    return {
        "session_id": session_id,
        "user_message": request.message,
        "assistant_response": response,
        "agent_type": agent_type
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint — returns tokens as they arrive."""
    session_id, agent = get_or_create_session(request.session_id)

    async def token_generator():
        # Get full response
        response = agent.chat(request.message)

        # Stream word by word
        words = response.split(" ")
        for word in words:
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.03)  # Small delay between tokens
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session_id}
    )


@app.get("/session/{session_id}/memory")
async def get_memory(session_id: str):
    """Get session memory — profile + history."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id].get_memory_summary()


@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id].reset()
    return {"message": "Short-term memory reset. Long-term memory preserved."}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted."}
