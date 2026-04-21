# 🎬 AutoStream AI Agent — Social-to-Lead Agentic Workflow

> **Machine Learning Intern Assignment | ServiceHive × Inflx**  
> A production-grade Conversational AI Agent that converts social media conversations into qualified business leads.

---

## 📌 Project Overview

This project implements a **Conversational AI Agent** for **AutoStream** — a fictional SaaS platform offering automated video editing tools for content creators. The agent is built using **LangGraph + Gemini 1.5 Flash** and demonstrates:

- ✅ Intent classification (greeting / product inquiry / high intent)
- ✅ RAG-powered knowledge retrieval from a local JSON knowledge base
- ✅ Multi-turn state management across 5–6 conversation turns
- ✅ Controlled, sequential lead data collection (Name → Email → Platform)
- ✅ Mock CRM tool execution only after all fields are confirmed

---

## 🗂️ Project Structure

```
autostream-agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py              # LangGraph nodes, edges, routing logic
│   ├── intent_classifier.py  # Rule-based intent + field extractor
│   └── state.py              # TypedDict state schema
├── knowledge_base/
│   └── autostream_kb.json    # Local RAG knowledge base
├── tools/
│   ├── __init__.py
│   ├── lead_capture.py       # mock_lead_capture() tool
│   └── rag_retriever.py      # Context retrieval from KB
├── main.py                   # CLI entry point
├── leads.json                # Auto-generated: captured leads log
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
```bash
cp .env.example .env
# Open .env and paste your Gemini API key
# Get it free at: https://aistudio.google.com/app/apikey
```

### 5. Run the Agent
```bash
python main.py
```

---

## 💬 Example Conversation

```
You: Hi, tell me about your pricing.
Aria: AutoStream offers two plans — Basic at $29/month (10 videos, 720p) and 
      Pro at $79/month (unlimited videos, 4K, AI captions). Which fits your needs?

You: That sounds good, I want to try the Pro plan for my YouTube channel.
Aria: Awesome choice! I'd love to get you set up. What's your name?

You: Priya Sharma
Aria: Nice to meet you, Priya! What's your email address?

You: priya@example.com
Aria: Perfect! And which platform do you mainly create on?

You: YouTube
Aria: Perfect! Let me get you set up.

✅  LEAD CAPTURED SUCCESSFULLY
  Lead ID  : LEAD-20250621143022
  Name     : Priya Sharma
  Email    : priya@example.com
  Platform : YouTube

Aria: 🎉 You're all set, Priya! Our team will reach out to priya@example.com 
      within 24 hours with your onboarding details.
```

---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it offers **explicit, deterministic state machines** — critical for a sales agent where sequence matters. Unlike AutoGen's multi-agent chat loop (which can be unpredictable for sequential form collection), LangGraph lets you define exactly which node runs under which condition. This prevents the agent from jumping ahead to email collection before capturing the user's name, or triggering the lead capture tool prematurely.

### State Management

State is managed using a **TypedDict (`AgentState`)** that persists across all nodes in the graph. Every conversation turn carries forward: the full message history, classified intent, partial lead fields (`lead_name`, `lead_email`, `lead_platform`), a boolean `collecting_lead` flag, and the current RAG context. LangGraph's `add_messages` annotation ensures message history is append-only and never overwritten.

### RAG Pipeline

The local JSON knowledge base is queried using **keyword-based retrieval** — scanning the user's message for pricing, policy, or FAQ-related terms and injecting matching sections into the LLM system prompt. This simulates a lightweight vector search without requiring an embedding model, making it zero-cost and fully offline.

### Tool Execution Safety

The `mock_lead_capture()` tool is placed in a dedicated `capture_lead` node that is only reachable once all three lead fields are non-null in the state. It cannot be triggered prematurely by any routing path.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following architecture would be used:

### Architecture

```
WhatsApp User
     │
     ▼
[WhatsApp Business API / Twilio for WhatsApp]
     │  (HTTP POST Webhook)
     ▼
[FastAPI / Flask Webhook Server]   ←── receives message
     │
     ├── Validates webhook signature (security)
     ├── Extracts sender number + message text
     ├── Looks up or creates AgentState for this phone number
     │        (stored in Redis or a DB keyed by phone number)
     │
     ▼
[autostream LangGraph Agent]
     │  (invoke with restored state)
     ▼
Agent Response
     │
     ▼
[WhatsApp Business API]  ─── sends reply back to user
```

### Step-by-Step Integration

1. **Set up WhatsApp Business API**  
   Use [Twilio](https://www.twilio.com/whatsapp) or Meta's [Cloud API](https://developers.facebook.com/docs/whatsapp/cloud-api). Register a business number.

2. **Create a Webhook Endpoint**
   ```python
   # webhook.py (FastAPI example)
   from fastapi import FastAPI, Request
   import json, redis
   from agent.graph import agent_graph
   
   app = FastAPI()
   r = redis.Redis()
   
   @app.post("/webhook")
   async def whatsapp_webhook(request: Request):
       data = await request.json()
       sender = data["From"]         # e.g., "whatsapp:+91XXXXXXXXXX"
       message = data["Body"]        # User's text message
       
       # Restore or initialize state per user
       raw_state = r.get(sender)
       state = json.loads(raw_state) if raw_state else initialize_state()
       
       # Run the agent
       response, new_state = run_agent(message, state)
       
       # Persist state back to Redis
       r.setex(sender, 86400, json.dumps(new_state, default=str))  # TTL: 24h
       
       # Reply via Twilio/WhatsApp API
       send_whatsapp_message(to=sender, body=response)
       return {"status": "ok"}
   ```

3. **State Persistence**  
   Use **Redis** (with TTL) to store each user's `AgentState` keyed by phone number. This enables multi-turn conversations across separate webhook calls.

4. **Register Webhook URL with Meta/Twilio**  
   Point the webhook to `https://your-domain.com/webhook`. WhatsApp will POST each incoming message there.

5. **Security**  
   Validate the `X-Hub-Signature-256` header on every incoming request to ensure it's genuinely from Meta/Twilio.

### Key Considerations
- **Session TTL**: Set Redis keys to expire after 24–48 hours of inactivity
- **Rate Limiting**: WhatsApp has message-per-second limits; add a queue (Celery + Redis) for high volume
- **Opt-in compliance**: Users must opt in to receive messages per WhatsApp Business Policy
- **Media handling**: Extend the agent to handle voice notes → transcribe → process (Whisper integration)

---

## 🔑 Getting a Free Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy and paste it into your `.env` file

---

## 📊 Evaluation Checklist

| Criteria | Implementation |
|---|---|
| Agent Reasoning & Intent Detection | `agent/intent_classifier.py` — regex patterns + state-aware routing |
| RAG Pipeline | `tools/rag_retriever.py` — keyword retrieval from local JSON KB |
| State Management | `agent/state.py` — TypedDict + LangGraph `add_messages` |
| Tool Calling Logic | `node_capture_lead` — only fires when all 3 fields are non-null |
| Code Clarity | Modular structure, docstrings, type hints throughout |
| Real-world Deployability | WhatsApp webhook architecture documented above |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph 0.4+ |
| LLM | Gemini 1.5 Flash (Google) |
| RAG | Local JSON + keyword retrieval |
| State | LangGraph TypedDict + add_messages |
| Lead Tool | mock_lead_capture() → leads.json |

---

*Built for ServiceHive × Inflx ML Intern Assignment*
