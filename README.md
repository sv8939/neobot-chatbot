# NeoBot — AI Chatbot with RAG & Web Search

A Streamlit-based intelligent chatbot built for the NeoStats AI Engineer assessment.

## Features

| Feature | Description |
|---|---|
| 🧠 Multi-provider LLM | Groq, OpenAI, Google Gemini |
| 📚 RAG | Upload PDF / TXT / DOCX and chat with your documents |
| 🌐 Live Web Search | Real-time Tavily search for current information |
| 📝 Response Modes | Concise (2-4 sentences) or Detailed (full explanation) |
| 🎭 Custom Persona | Override the system prompt from the UI |

## Project Structure

```
project/
├── config/
│   └── config.py          ← All API keys and settings (env vars)
├── models/
│   ├── llm.py             ← LLM factory (Groq / OpenAI / Gemini)
│   └── embeddings.py      ← Embedding models (HuggingFace / OpenAI)
├── utils/
│   ├── rag_utils.py       ← Document ingestion & FAISS retrieval
│   ├── search_utils.py    ← Live web search via Tavily
│   └── prompt_utils.py    ← System prompt assembly
├── app.py                 ← Main Streamlit UI
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export GROQ_API_KEY=your_key
export TAVILY_API_KEY=your_key   # optional, for web search

# 3. Run
streamlit run app.py
```

## API Keys

- **Groq** (recommended, free tier): https://console.groq.com/keys
- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://aistudio.google.com/app/apikey
- **Tavily** (web search, free tier): https://tavily.com

## Architecture

```
User Message
     │
     ├─── RAG Retrieval (FAISS similarity search) ──┐
     │                                              │
     ├─── Web Search (Tavily, if triggered)  ───────┤
     │                                              │
     └───────────────────────────────────────── System Prompt
                                                    │
                                               LLM (Groq/OpenAI/Gemini)
                                                    │
                                              Assistant Reply
```
