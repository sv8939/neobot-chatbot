"""
models/llm.py — Factory functions to initialise supported chat-model providers.

Supported providers
-------------------
* Groq   – llama / mixtral family (default)
* OpenAI – GPT-4o / GPT-4o-mini
* Google – Gemini 1.5 flash / pro
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    GROQ_API_KEY, GROQ_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GOOGLE_API_KEY, GOOGLE_MODEL,
)


def get_chatgroq_model():
    """Return a LangChain ChatGroq instance."""
    try:
        from langchain_groq import ChatGroq
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set.")
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Groq model: {exc}") from exc


def get_openai_model():
    """Return a LangChain ChatOpenAI instance."""
    try:
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise OpenAI model: {exc}") from exc


def get_google_model():
    """Return a LangChain ChatGoogleGenerativeAI instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        return ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model=GOOGLE_MODEL)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Google Gemini model: {exc}") from exc


def get_model(provider: str = "groq"):
    """
    Convenience factory that picks the right model by provider name.

    Parameters
    ----------
    provider : str
        One of "groq", "openai", "google".

    Returns
    -------
    LangChain BaseChatModel
    """
    provider = provider.lower().strip()
    if provider == "groq":
        return get_chatgroq_model()
    if provider == "openai":
        return get_openai_model()
    if provider in ("google", "gemini"):
        return get_google_model()
    raise ValueError(f"Unknown provider '{provider}'. Choose: groq | openai | google.")
