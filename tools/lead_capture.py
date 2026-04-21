"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
"""

import json
import os
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture a qualified lead.
    Simulates writing to a CRM or backend database.
    
    Args:
        name: Full name of the lead
        email: Email address of the lead
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)
    
    Returns:
        dict with success status and lead ID
    """
    lead_id = f"LEAD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    print("\n" + "="*50)
    print("✅  LEAD CAPTURED SUCCESSFULLY")
    print("="*50)
    print(f"  Lead ID  : {lead_id}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    # Optionally persist to a local JSON file (simulating CRM storage)
    leads_file = os.path.join(os.path.dirname(__file__), "..", "leads.json")
    leads = []
    if os.path.exists(leads_file):
        with open(leads_file, "r") as f:
            leads = json.load(f)
    
    leads.append({
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.now().isoformat()
    })
    
    with open(leads_file, "w") as f:
        json.dump(leads, f, indent=2)
    
    return {
        "success": True,
        "lead_id": lead_id,
        "message": f"Lead captured successfully: {name}, {email}, {platform}"
    }
