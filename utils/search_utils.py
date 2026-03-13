"""
utils/search_utils.py — Live web search integration via Tavily.

Tavily is purpose-built for LLM agents and returns clean, structured results.
Sign up for a free API key at https://tavily.com.

If Tavily is unavailable, the module gracefully returns an empty result
so the chatbot degrades to LLM-only mode without crashing.
"""

import sys, os, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 3) -> str:
    """
    Perform a live web search for the given query using Tavily.

    Parameters
    ----------
    query : str
        The search query derived from the user's message.
    max_results : int
        Number of results to fetch (1-5 recommended).

    Returns
    -------
    str
        Formatted search results suitable for injection into a system prompt,
        or an empty string if the search fails or no key is configured.
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set; web search disabled.")
        return ""

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results, search_depth="basic")

        results = response.get("results", [])
        if not results:
            return ""

        lines = ["[Live Web Search Results]"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "").strip()
            lines.append(f"\n{i}. {title}\n   URL: {url}\n   {content[:400]}")

        return "\n".join(lines)

    except ImportError:
        logger.error("tavily-python not installed. Run: pip install tavily-python")
        return ""
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return ""


def should_search(user_message: str) -> bool:
    """
    Heuristic to decide whether a web search would help answer the query.
    Returns True for questions that imply need for current/external information.
    """
    triggers = [
        "latest", "recent", "today", "current", "news", "update",
        "price", "stock", "weather", "who is", "what is", "when did",
        "how much", "compare", "vs", "2024", "2025", "search",
    ]
    lower = user_message.lower()
    return any(t in lower for t in triggers)
