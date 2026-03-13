"""
models/embeddings.py — Embedding model factory for RAG.

Uses HuggingFace sentence-transformers by default (no API key needed).
Falls back to OpenAI embeddings if OPENAI_API_KEY is set and requested.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import EMBEDDING_MODEL, OPENAI_API_KEY


def get_hf_embeddings():
    """
    Return a HuggingFaceEmbeddings instance using the configured sentence-transformer.
    Works fully offline after the first model download.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load HuggingFace embeddings: {exc}") from exc


def get_openai_embeddings():
    """Return OpenAI text-embedding-3-small embeddings (requires OPENAI_API_KEY)."""
    try:
        from langchain_openai import OpenAIEmbeddings
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    except Exception as exc:
        raise RuntimeError(f"Failed to load OpenAI embeddings: {exc}") from exc


def get_embeddings(provider: str = "huggingface"):
    """
    Factory for embedding models.

    Parameters
    ----------
    provider : str
        "huggingface" (default, free) or "openai".
    """
    provider = provider.lower().strip()
    if provider == "huggingface":
        return get_hf_embeddings()
    if provider == "openai":
        return get_openai_embeddings()
    raise ValueError(f"Unknown embedding provider '{provider}'. Choose: huggingface | openai.")
