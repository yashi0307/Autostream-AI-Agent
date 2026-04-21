# AutoStream AI Agent — Social-to-Lead Workflow

Machine Learning Intern Assignment — ServiceHive × Inflx

---

## Overview

This project is a conversational AI agent built for a fictional SaaS product called **AutoStream**, which provides automated video editing tools for content creators.

The goal of the agent is not just to chat, but to behave like a **sales assistant** — it understands user intent, answers product-related questions using a knowledge base, identifies users who are ready to sign up, and collects their details as leads.

The system is designed to simulate how real-world AI agents convert conversations into business outcomes.

---

## What the Agent Can Do

* Classify user intent into:

  * Greeting
  * Product inquiry
  * High-intent (ready to sign up)

* Answer questions using a local knowledge base (RAG)

* Maintain conversation context across multiple turns

* Collect lead details step-by-step:

  * Name
  * Email
  * Platform (YouTube, Instagram, etc.)

* Trigger a backend tool only after all required details are collected

---

## Project Structure

```text
autostream-agent/
├── agent/
│   ├── graph.py
│   ├── intent_classifier.py
│   └── state.py
├── knowledge_base/
│   └── autostream_kb.json
├── tools/
│   ├── lead_capture.py
│   └── rag_retriever.py
├── main.py
├── leads.json
├── requirements.txt
├── .env.example
└── README.md
```

---
## Demo Video

Add your demo video link here:

```
https://drive.google.com/file/d/16sdvVSMwmUQfJBLSdy-CPVrMTiC3XHAv/view?usp=sharing
```

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/autostream-ai-agent.git
cd autostream-agent
```

2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Add your API key
   Create a `.env` file and add:

```bash
GOOGLE_API_KEY=your_api_key_here
```

5. Run the project

```bash
python main.py
```

---

## Example Flow

User asks about pricing → agent responds using knowledge base

User shows interest → agent detects high intent

Agent collects:

* Name
* Email
* Platform

Then triggers:

```
Lead captured successfully: Name, Email, Platform
```

---

## Architecture (Why This Approach)

This project uses **LangGraph** to control the flow of the agent.

Instead of a simple chatbot, the system is designed as a structured workflow where each step is controlled. This ensures that:

* The agent does not skip steps (like asking for email before name)
* The lead capture tool is only triggered after all required data is collected
* The conversation remains predictable and consistent

State is maintained across turns using a shared state object, which stores:

* Messages
* Intent
* Collected user details

For answering questions, a local JSON knowledge base is used. Relevant information is retrieved and passed to the LLM, ensuring responses are grounded and not hallucinated.


---

## WhatsApp Integration (Approach)

To deploy this agent on WhatsApp, it can be integrated using the Twilio WhatsApp API.

Basic flow:

1. User sends a message on WhatsApp
2. Twilio forwards it to a backend webhook (Flask/FastAPI)
3. The backend sends the message to the agent
4. The agent processes it and generates a response
5. Response is sent back to the user via Twilio

User state can be stored using Redis or a database, keyed by phone number, so conversations continue across messages.

---

## Tech Stack

* Python 3.9+
* LangGraph
* Gemini 1.5 Flash
* JSON-based knowledge base
* FAISS / retrieval logic
* Flask / FastAPI (for deployment)

---


## What This Project Demonstrates

* Intent-aware conversational design
* Use of RAG for reliable responses
* Multi-turn state management
* Controlled tool execution
* Real-world deployable architecture

---

## Final Note

The focus of this project was to build something closer to a **real AI product workflow** rather than a basic chatbot. The agent is designed to guide users from conversation to conversion in a structured way.

---

