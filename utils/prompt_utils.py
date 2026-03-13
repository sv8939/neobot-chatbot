"""
utils/prompt_utils.py — Prompt assembly helpers.

Handles merging of:
* Base system persona
* RAG context (retrieved document excerpts)
* Web search results
* Response-mode suffix (concise vs detailed)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import CONCISE_PROMPT_SUFFIX, DETAILED_PROMPT_SUFFIX

BASE_SYSTEM_PROMPT = """You are NeoBot, an intelligent, helpful AI assistant.
You answer questions accurately, cite your sources when using retrieved context,
and acknowledge when you are uncertain rather than guessing."""


def build_system_prompt(
    rag_context: str = "",
    search_results: str = "",
    response_mode: str = "detailed",
    custom_persona: str = "",
) -> str:
    """
    Assemble the full system prompt from components.

    Parameters
    ----------
    rag_context : str
        Document excerpts retrieved by the RAG pipeline.
    search_results : str
        Live web search results (may be empty).
    response_mode : str
        "concise" or "detailed".
    custom_persona : str
        Optional user-supplied system persona override.

    Returns
    -------
    str  — the complete system prompt to pass to the LLM.
    """
    parts = [custom_persona.strip() if custom_persona.strip() else BASE_SYSTEM_PROMPT]

    if rag_context:
        parts.append(
            "\n\n## Knowledge Base Context\n"
            "Use the following excerpts from uploaded documents to inform your answer. "
            "Always specify which document the information comes from.\n\n"
            + rag_context
        )

    if search_results:
        parts.append(
            "\n\n## Live Web Search Results\n"
            "The following real-time information was retrieved from the web. "
            "Incorporate it where relevant and cite the source URL.\n\n"
            + search_results
        )

    # Response-mode suffix
    if response_mode == "concise":
        parts.append(CONCISE_PROMPT_SUFFIX)
    else:
        parts.append(DETAILED_PROMPT_SUFFIX)

    return "".join(parts)
