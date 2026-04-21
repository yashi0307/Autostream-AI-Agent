"""
tools/rag_retriever.py
Local RAG (Retrieval-Augmented Generation) pipeline for AutoStream knowledge base.
Uses simple keyword + semantic matching over the local JSON knowledge base.
"""

import json
import os
import re
from typing import Optional


KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "autostream_kb.json")


def load_knowledge_base() -> dict:
    """Load the AutoStream knowledge base from JSON."""
    with open(KB_PATH, "r") as f:
        return json.load(f)


def format_pricing_context(kb: dict) -> str:
    """Format pricing data into readable text for LLM context."""
    plans = kb["pricing"]["plans"]
    lines = ["AutoStream Pricing Plans:\n"]
    for plan in plans:
        lines.append(f"### {plan['name']} — ${plan['price_monthly']}/month")
        lines.append(f"Best for: {plan['best_for']}")
        lines.append("Features:")
        for feat in plan["features"]:
            lines.append(f"  - {feat}")
        lines.append("")
    return "\n".join(lines)


def format_policies_context(kb: dict) -> str:
    """Format policy data into readable text for LLM context."""
    p = kb["policies"]
    return f"""AutoStream Policies:
- Refund Policy: {p['refund_policy']}
- Cancellation: {p['cancellation']}
- Free Trial: {p['free_trial']}
- Support (Basic Plan): {p['support']['basic_plan']}
- Support (Pro Plan): {p['support']['pro_plan']}
"""


def format_faqs_context(kb: dict) -> str:
    """Format FAQs into readable text."""
    lines = ["AutoStream FAQs:\n"]
    for faq in kb["faqs"]:
        lines.append(f"Q: {faq['question']}")
        lines.append(f"A: {faq['answer']}\n")
    return "\n".join(lines)


def retrieve_context(query: str) -> str:
    """
    Retrieve relevant knowledge base context based on the user query.
    Uses keyword matching to select the most relevant sections.
    
    Args:
        query: The user's query string
    
    Returns:
        Concatenated relevant context string for the LLM
    """
    kb = load_knowledge_base()
    query_lower = query.lower()
    
    context_parts = []
    
    # Always include company overview
    company = kb["company"]
    context_parts.append(
        f"Company: {company['name']} — {company['tagline']}\n{company['description']}\n"
    )
    
    # Pricing keywords
    pricing_keywords = ["price", "pricing", "plan", "cost", "dollar", "$", "basic", "pro",
                        "how much", "subscription", "monthly", "annual", "4k", "720p",
                        "unlimited", "videos", "resolution", "caption", "feature"]
    if any(kw in query_lower for kw in pricing_keywords):
        context_parts.append(format_pricing_context(kb))
    
    # Policy keywords
    policy_keywords = ["refund", "cancel", "support", "policy", "money back",
                       "24/7", "guarantee", "trial", "free"]
    if any(kw in query_lower for kw in policy_keywords):
        context_parts.append(format_policies_context(kb))
    
    # FAQ keywords
    faq_keywords = ["youtube", "instagram", "tiktok", "platform", "upgrade", "team",
                    "export", "annual", "discount", "enterprise"]
    if any(kw in query_lower for kw in faq_keywords):
        context_parts.append(format_faqs_context(kb))
    
    # If nothing matched beyond company info, include everything
    if len(context_parts) == 1:
        context_parts.append(format_pricing_context(kb))
        context_parts.append(format_policies_context(kb))
        context_parts.append(format_faqs_context(kb))
    
    return "\n---\n".join(context_parts)
