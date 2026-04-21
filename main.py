"""
main.py
AutoStream Social-to-Lead Agent — CLI Entry Point.
Run: python main.py
"""

import sys
import os
from langchain_core.messages import HumanMessage

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import agent_graph
from agent.state import AgentState


def print_banner():
    print("\n" + "="*60)
    print("  🎬  AutoStream AI Agent  —  Powered by Gemini 1.5 Flash")
    print("  Built with LangGraph | RAG | Intent Detection")
    print("="*60)
    print("  Type your message and press Enter.")
    print("  Type 'quit' or 'exit' to end the session.\n")


def initialize_state() -> AgentState:
    """Create a fresh agent state."""
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


def run_agent(user_input: str, state: AgentState) -> tuple[str, AgentState]:
    """
    Run one turn of the agent.
    
    Args:
        user_input: Raw user message string
        state: Current conversation state
    
    Returns:
        (agent_response, updated_state)
    """
    # Add user message to state
    updated_state = {
        **state,
        "messages": state["messages"] + [HumanMessage(content=user_input)],
    }
    
    # Run the LangGraph agent
    result_state = agent_graph.invoke(updated_state)
    
    # Extract the latest AI response
    from langchain_core.messages import AIMessage
    ai_messages = [m for m in result_state["messages"] if isinstance(m, AIMessage)]
    latest_response = ai_messages[-1].content if ai_messages else "I'm not sure how to respond to that."
    
    return latest_response, result_state


def main():
    print_banner()
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌  ERROR: GOOGLE_API_KEY not found.")
        print("   Create a .env file with: GOOGLE_API_KEY=your_key_here")
        print("   Get your key at: https://aistudio.google.com/app/apikey\n")
        sys.exit(1)
    
    state = initialize_state()
    
    print("Aria (AutoStream AI): Hi there! 👋 I'm Aria, your AutoStream assistant.")
    print("  I can help with pricing, features, and getting you started. What brings you here today?\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAria: Thanks for chatting! Have a great day. 🎬\n")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("\nAria: Thanks for chatting with AutoStream! 🎬 See you around!\n")
            break
        
        try:
            response, state = run_agent(user_input, state)
            print(f"\nAria: {response}\n")
            
            # Show intent debug info (optional — remove for production)
            intent = state.get("intent", "unknown")
            collecting = state.get("collecting_lead", False)
            captured = state.get("lead_captured", False)
            
            debug_info = f"[Debug → Intent: {intent}"
            if collecting:
                name = state.get("lead_name") or "?"
                email = state.get("lead_email") or "?"
                platform = state.get("lead_platform") or "?"
                debug_info += f" | Collecting: name={name}, email={email}, platform={platform}"
            if captured:
                debug_info += " | ✅ Lead Captured"
            debug_info += "]"
            print(f"\033[90m{debug_info}\033[0m\n")
            
            # If lead is captured, offer to start a new session
            if captured:
                print("Aria: Is there anything else I can help you with?\n")
                # Reset lead collection state but keep conversation going
                state = {
                    **state,
                    "collecting_lead": False,
                }
                
        except Exception as e:
            print(f"\n❌ Agent error: {e}")
            print("   Please check your API key and try again.\n")


if __name__ == "__main__":
    main()
