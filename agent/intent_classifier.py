"""
agent/intent_classifier.py
Rule-based + LLM-assisted intent classifier for the AutoStream agent.
Classifies user messages into: greeting, product_inquiry, high_intent, lead_collection, unknown.
"""

import re
from typing import Literal

IntentType = Literal["greeting", "product_inquiry", "high_intent", "lead_collection", "unknown"]


# High-intent signal phrases
HIGH_INTENT_PATTERNS = [
    r"\b(sign up|signup|subscribe|buy|purchase|get started|start now|ready to|want to try|let's go|i'm in|i am in)\b",
    r"\b(upgrade|switch to pro|go pro|pro plan|take the pro|basic plan|get the)\b",
    r"\b(my (youtube|instagram|tiktok|channel|page|account))\b",
    r"\b(how do i (sign|get|start|join))\b",
    r"\b(interested in (pro|basic|plan|subscribing|buying))\b",
    r"\b(i want (to try|the|to get|to sign|to subscribe))\b",
    r"\b(where do i (sign|pay|subscribe|buy|start))\b",
    r"\b(can i (try|get|have|join|start))\b",
]

# Product/pricing inquiry patterns
PRODUCT_INQUIRY_PATTERNS = [
    r"\b(price|pricing|cost|how much|plan|plans|feature|features|what do|what does|include|offer|difference)\b",
    r"\b(basic|pro|4k|720p|resolution|captions|videos|unlimited|refund|support|trial|cancel|policy)\b",
    r"\b(tell me|explain|describe|what is|what are|do you have|do you offer)\b",
    r"\b(compare|vs|versus|better|which plan)\b",
]

# Greeting patterns
GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|howdy|what's up|sup|greetings|yo)\b",
    r"^\s*(hi there|hello there|hey there)\b",
]

# Lead collection info patterns (user is providing their details)
LEAD_COLLECTION_PATTERNS = [
    r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",  # email
    r"\b(my name is|i am|i'm|name:|email:|platform:)\b",
    r"\b(youtube|instagram|tiktok|facebook|twitter|linkedin)\b",
]


def classify_intent(user_message: str, currently_collecting: bool = False) -> IntentType:
    """
    Classify the intent of a user message.
    
    Priority order:
    1. If actively collecting lead info → lead_collection
    2. High-intent signals → high_intent
    3. Product/pricing inquiry → product_inquiry
    4. Greeting → greeting
    5. Fallback → unknown
    
    Args:
        user_message: The raw user message string
        currently_collecting: Whether we're mid lead-collection flow
    
    Returns:
        IntentType classification
    """
    msg = user_message.lower().strip()
    
    # If we're already collecting lead info, treat as lead collection
    if currently_collecting:
        return "lead_collection"
    
    # Check high-intent first (stronger signal)
    for pattern in HIGH_INTENT_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return "high_intent"
    
    # Check product/pricing inquiry
    for pattern in PRODUCT_INQUIRY_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return "product_inquiry"
    
    # Check greeting
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return "greeting"
    
    return "unknown"


def extract_lead_field(message: str, field: str) -> str | None:
    """
    Try to extract a specific lead field from a free-text message.
    
    Args:
        message: User's message
        field: One of 'email', 'name', 'platform'
    
    Returns:
        Extracted value or None
    """
    msg = message.strip()
    
    if field == "email":
        match = re.search(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b", msg)
        return match.group(0) if match else None
    
    elif field == "platform":
        platforms = ["youtube", "instagram", "tiktok", "facebook", "twitter", "linkedin", "snapchat", "pinterest"]
        for p in platforms:
            if p in msg.lower():
                return p.capitalize()
        return None
    
    elif field == "name":
        # Try "my name is X" or "I'm X" or "I am X"
        patterns = [
            r"my name is ([A-Za-z\s]+)",
            r"i'?m ([A-Za-z\s]+)",
            r"i am ([A-Za-z\s]+)",
            r"name[:\s]+([A-Za-z\s]+)",
            r"^([A-Za-z]{2,}\s[A-Za-z]{2,})",  # "First Last" at start
        ]
        for pat in patterns:
            match = re.search(pat, msg, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out common false positives
                if name.lower() not in ["interested", "a youtuber", "a creator", "on youtube"]:
                    return name.title()
        return None
    
    return None
