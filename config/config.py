"""
config.py — Centralised configuration for API keys and model settings.
All secrets are loaded from environment variables; never commit real keys.
"""
from dotenv import load_dotenv
load_dotenv()

import os

# ── LLM Provider Keys ────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Web Search ───────────────────────────────────────────────────────────────
# Tavily is the recommended search provider for LangChain agents.
# Sign up free at https://tavily.com
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── Model Defaults ───────────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "groq")

GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

# ── RAG / Embeddings ─────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))

# ── Response Modes ────────────────────────────────────────────────────────────
CONCISE_PROMPT_SUFFIX: str = (
    "\n\nIMPORTANT: Keep your reply SHORT and CONCISE — ideally 2-4 sentences. "
    "Prioritise the key answer; skip preamble."
)
DETAILED_PROMPT_SUFFIX: str = (
    "\n\nIMPORTANT: Give a DETAILED, thorough answer. Explain your reasoning, "
    "include examples where helpful, and structure your response clearly."
)
