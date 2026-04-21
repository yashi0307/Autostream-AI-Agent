"""
server.py
FastAPI backend that exposes the AutoStream LangGraph agent via HTTP API.
Run: uvicorn server:app --reload --port 8000
"""

import os
import json
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import agent_graph
from agent.state import AgentState

load_dotenv()

app = FastAPI(title="AutoStream AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (keyed by session_id)
sessions: dict[str, AgentState] = {}


def init_state() -> AgentState:
    return AgentState(
        messages=[],
        intent=None,
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        collecting_lead=False,
        rag_context=None,
        turn_count=0,
    )


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: Optional[str]
    lead_captured: bool
    collecting_lead: bool
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Create or restore session
    session_id = req.session_id or str(uuid.uuid4())
    state = sessions.get(session_id, init_state())

    # Add user message
    state = {**state, "messages": state["messages"] + [HumanMessage(content=req.message)]}

    try:
        result = agent_graph.invoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract latest AI reply
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = ai_msgs[-1].content if ai_msgs else "I'm not sure how to respond to that."

    # Persist updated state
    sessions[session_id] = result

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        intent=result.get("intent"),
        lead_captured=result.get("lead_captured", False),
        collecting_lead=result.get("collecting_lead", False),
        lead_name=result.get("lead_name"),
        lead_email=result.get("lead_email"),
        lead_platform=result.get("lead_platform"),
    )


@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "reset"}


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "AutoStream AI"}


# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
