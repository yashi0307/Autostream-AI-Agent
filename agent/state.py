"""
agent/state.py
LangGraph state schema for the AutoStream Social-to-Lead Agent.
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# Intent classification types
IntentType = Literal["greeting", "product_inquiry", "high_intent", "lead_collection", "unknown"]


class AgentState(TypedDict):
    """
    Full conversation state managed across LangGraph nodes.
    
    Fields:
        messages: Full conversation history (auto-merged by LangGraph)
        intent: Current classified intent of the latest user message
        lead_name: Collected lead name (None until provided)
        lead_email: Collected lead email (None until provided)
        lead_platform: Collected lead platform (None until provided)
        lead_captured: Whether mock_lead_capture has been called
        collecting_lead: Whether we are actively in lead collection flow
        rag_context: Retrieved RAG context for the current turn
        turn_count: Number of conversation turns completed
    """
    messages: Annotated[list, add_messages]
    intent: Optional[IntentType]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool
    rag_context: Optional[str]
    turn_count: int
