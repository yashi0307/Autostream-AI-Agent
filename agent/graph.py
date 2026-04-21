"""
agent/graph.py
LangGraph-based Conversational AI Agent for AutoStream.
Implements: Intent Detection → RAG Retrieval → Lead Collection → Tool Execution
"""

import os
from typing import Literal
from dotenv import load_dotenv


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.intent_classifier import classify_intent, extract_lead_field
from tools.rag_retriever import retrieve_context
from tools.lead_capture import mock_lead_capture

load_dotenv()

# ─────────────────────────────────────────────
# LLM Initialization (Gemini 1.5 Flash)
# ─────────────────────────────────────────────

from langchain_groq import ChatGroq


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.7,
    )


# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are Aria, the friendly and knowledgeable AI sales assistant for AutoStream — 
a SaaS platform that provides automated video editing tools for content creators.

Your responsibilities:
1. Warmly greet users and understand their needs.
2. Answer product and pricing questions ONLY using the provided knowledge base context.
3. Identify when users show high purchase intent and transition smoothly to lead collection.
4. Collect: Name, Email, and Creator Platform — one at a time, conversationally.
5. Confirm before finalizing lead capture.

Tone: Friendly, concise, enthusiastic about AutoStream. Never pushy.
Rules:
- Only answer questions based on the provided context. Do not make up features or prices.
- Never ask for all three lead fields at once — collect them one by one naturally.
- Once you have all three lead fields, say: "Perfect! Let me get you set up." (this triggers backend capture).
- Keep responses under 100 words unless explaining pricing in detail.
"""


# ─────────────────────────────────────────────
# Node: Classify Intent
# ─────────────────────────────────────────────

def node_classify_intent(state: AgentState) -> AgentState:
    """Classify the intent of the latest user message."""
    messages = state["messages"]
    last_user_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    
    intent = classify_intent(last_user_msg, currently_collecting=state.get("collecting_lead", False))
    
    return {
        **state,
        "intent": intent,
        "turn_count": state.get("turn_count", 0) + 1,
    }


# ─────────────────────────────────────────────
# Node: RAG Retrieval
# ─────────────────────────────────────────────

def node_rag_retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant knowledge base context for the current query."""
    messages = state["messages"]
    last_user_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    
    context = retrieve_context(last_user_msg)
    return {**state, "rag_context": context}


# ─────────────────────────────────────────────
# Node: Generate Response (Greeting / General)
# ─────────────────────────────────────────────

def node_respond(state: AgentState) -> AgentState:
    """Generate an LLM response for greetings and general queries."""
    llm = get_llm()
    context = state.get("rag_context", "")
    
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\n[Knowledge Base Context]\n{context}"
    
    # Build message list for LLM (exclude SystemMessage from history)
    chat_history = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    
    full_messages = [SystemMessage(content=system_content)] + chat_history
    response = llm.invoke(full_messages)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


# ─────────────────────────────────────────────
# Node: Handle High Intent
# ─────────────────────────────────────────────

def node_high_intent(state: AgentState) -> AgentState:
    """
    User shows purchase intent. Start lead collection by asking for name.
    Attempt to extract any info already mentioned in this message.
    """
    llm = get_llm()
    messages = state["messages"]
    last_user_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    
    # Try to pre-extract info from intent message
    lead_name = state.get("lead_name") or extract_lead_field(last_user_msg, "name")
    lead_email = state.get("lead_email") or extract_lead_field(last_user_msg, "email")
    lead_platform = state.get("lead_platform") or extract_lead_field(last_user_msg, "platform")
    
    context = state.get("rag_context", "")
    system_content = SYSTEM_PROMPT
    if context:
        system_content += f"\n\n[Knowledge Base Context]\n{context}"
    
    # Tell LLM what we've collected so far
    system_content += f"""
\n[Lead Collection Status]
- Name collected: {lead_name or 'NO — ask for it first'}
- Email collected: {lead_email or 'NO — ask after name'}
- Platform collected: {lead_platform or 'NO — ask after email'}

The user just showed HIGH PURCHASE INTENT. Start collecting missing info ONE field at a time.
If name is missing, ask for their name warmly. Do not ask for email yet.
"""
    
    chat_history = [m for m in messages if not isinstance(m, SystemMessage)]
    full_messages = [SystemMessage(content=system_content)] + chat_history
    response = llm.invoke(full_messages)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "collecting_lead": True,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
    }


# ─────────────────────────────────────────────
# Node: Collect Lead Info
# ─────────────────────────────────────────────

def node_collect_lead(state: AgentState) -> AgentState:
    """
    Sequentially collect Name → Email → Platform from the user.
    Only moves to next field after current one is confirmed.
    """
    llm = get_llm()
    messages = state["messages"]
    last_user_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    
    lead_name = state.get("lead_name")
    lead_email = state.get("lead_email")
    lead_platform = state.get("lead_platform")
    
    # Try to extract the missing field from current message
    if not lead_name:
        lead_name = extract_lead_field(last_user_msg, "name") or last_user_msg.strip().title()
    elif not lead_email:
        lead_email = extract_lead_field(last_user_msg, "email")
        # If no email found in structured format, use raw message if it looks like email
        if not lead_email and "@" in last_user_msg:
            lead_email = last_user_msg.strip()
    elif not lead_platform:
        lead_platform = extract_lead_field(last_user_msg, "platform") or last_user_msg.strip().capitalize()
    
    system_content = SYSTEM_PROMPT + f"""
\n[Lead Collection Status]
- Name: {lead_name or 'MISSING'}
- Email: {lead_email or 'MISSING'}
- Platform: {lead_platform or 'MISSING'}

Rules:
- If name is collected but email is missing → ask for their email address naturally.
- If email is collected but platform is missing → ask which platform they create on.
- If ALL THREE are collected → respond with exactly: "Perfect! Let me get you set up." 
  Do not say anything else.
- Collect ONE field at a time. Be warm and conversational.
"""
    
    chat_history = [m for m in messages if not isinstance(m, SystemMessage)]
    full_messages = [SystemMessage(content=system_content)] + chat_history
    response = llm.invoke(full_messages)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
        "collecting_lead": True,
    }


# ─────────────────────────────────────────────
# Node: Execute Lead Capture Tool
# ─────────────────────────────────────────────

def node_capture_lead(state: AgentState) -> AgentState:
    """Execute the mock_lead_capture tool once all fields are collected."""
    name = state.get("lead_name", "Unknown")
    email = state.get("lead_email", "unknown@example.com")
    platform = state.get("lead_platform", "Unknown")
    
    result = mock_lead_capture(name, email, platform)
    
    confirmation_message = (
        f"🎉 You're all set, {name}! I've registered your interest in AutoStream Pro. "
        f"Our team will reach out to {email} within 24 hours with your onboarding details. "
        f"Can't wait to see what you create on {platform}!"
    )
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=confirmation_message)],
        "lead_captured": True,
        "collecting_lead": False,
    }


# ─────────────────────────────────────────────
# Routing Logic
# ─────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """Route to the correct node based on classified intent."""
    intent = state.get("intent", "unknown")
    collecting = state.get("collecting_lead", False)
    
    if collecting:
        return "collect_lead"
    
    if intent == "high_intent":
        return "rag_retrieve_then_high_intent"
    elif intent in ("product_inquiry", "unknown"):
        return "rag_retrieve"
    else:  # greeting
        return "respond"


def route_after_collect(state: AgentState) -> str:
    """Check if all lead fields are collected to trigger tool."""
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")
    captured = state.get("lead_captured", False)
    
    if not captured and name and email and platform:
        return "capture_lead"
    return END


def route_after_high_intent(state: AgentState) -> str:
    """After high-intent response, decide if we should immediately capture."""
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")
    
    if name and email and platform:
        return "capture_lead"
    return END


# ─────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """Construct and compile the LangGraph agent."""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("classify_intent", node_classify_intent)
    graph.add_node("rag_retrieve", node_rag_retrieve)
    graph.add_node("respond", node_respond)
    graph.add_node("high_intent", node_high_intent)
    graph.add_node("collect_lead", node_collect_lead)
    graph.add_node("capture_lead", node_capture_lead)
    
    # Entry point
    graph.set_entry_point("classify_intent")
    
    # Intent → routing
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "rag_retrieve": "rag_retrieve",
            "rag_retrieve_then_high_intent": "rag_retrieve",
            "collect_lead": "collect_lead",
            "respond": "respond",
        }
    )
    
    # After RAG: route to respond or high_intent
    graph.add_conditional_edges(
        "rag_retrieve",
        lambda state: "high_intent" if state.get("intent") == "high_intent" else "respond",
        {
            "high_intent": "high_intent",
            "respond": "respond",
        }
    )
    
    # After respond → END
    graph.add_edge("respond", END)
    
    # After high_intent → maybe capture or END
    graph.add_conditional_edges(
        "high_intent",
        route_after_high_intent,
        {
            "capture_lead": "capture_lead",
            END: END,
        }
    )
    
    # After collect_lead → maybe capture or END
    graph.add_conditional_edges(
        "collect_lead",
        route_after_collect,
        {
            "capture_lead": "capture_lead",
            END: END,
        }
    )
    
    # After capture → END
    graph.add_edge("capture_lead", END)
    
    return graph.compile()


# ─────────────────────────────────────────────
# Exported compiled graph
# ─────────────────────────────────────────────

agent_graph = build_agent_graph()
