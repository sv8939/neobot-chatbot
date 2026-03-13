"""
app.py — NeoBot: Intelligent Chatbot with RAG + Web Search
============================================================
Features
--------
* Multi-provider LLM support  (Groq, OpenAI, Google Gemini)
* RAG — upload PDF/TXT/DOCX and chat with your documents
* Live web search via Tavily when the LLM needs current data
* Response-mode toggle: Concise ↔ Detailed
* Clean Streamlit UI with persistent chat history per session

Run locally
-----------
    streamlit run app.py
"""

import os, sys, logging, tempfile
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Path setup (works when run from project root or AI_UseCase/) ──────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.llm import get_model
from models.embeddings import get_embeddings
from utils.rag_utils import ingest_documents, retrieve_context
from utils.search_utils import web_search, should_search
from utils.prompt_utils import build_system_prompt
from config.config import DEFAULT_LLM_PROVIDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeoBot — AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    .badge-rag  { background:#d1fae5; color:#065f46; }
    .badge-web  { background:#dbeafe; color:#1e40af; }
    .badge-llm  { background:#f3f4f6; color:#374151; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session-state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "doc_names": [],
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "response_mode": "detailed",
        "use_web_search": True,
        "custom_persona": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Botfather_icon.png/240px-Botfather_icon.png",
            width=60,
        )
        st.title("NeoBot ⚙️")
        st.caption("AI Chatbot · RAG · Web Search")
        st.divider()

        # ── LLM Provider ──────────────────────────────────────────────────────
        st.subheader("🧠 Language Model")
        provider = st.selectbox(
            "Provider",
            ["groq", "openai", "google"],
            index=["groq", "openai", "google"].index(st.session_state.llm_provider),
            key="provider_select",
        )
        st.session_state.llm_provider = provider

        # ── Response Mode ─────────────────────────────────────────────────────
        st.subheader("📝 Response Mode")
        mode = st.radio(
            "Mode",
            ["Concise", "Detailed"],
            index=0 if st.session_state.response_mode == "concise" else 1,
            horizontal=True,
        )
        st.session_state.response_mode = mode.lower()

        # ── Web Search Toggle ─────────────────────────────────────────────────
        st.subheader("🌐 Web Search")
        st.session_state.use_web_search = st.toggle(
            "Enable live web search", value=st.session_state.use_web_search
        )

        # ── RAG Document Upload ───────────────────────────────────────────────
        st.divider()
        st.subheader("📂 Knowledge Base (RAG)")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT, DOCX, MD)",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "md"],
        )

        if uploaded_files and st.button("🔍 Build Knowledge Base", use_container_width=True):
            with st.spinner("Embedding documents…"):
                try:
                    # Write uploads to a temp directory
                    tmp_dir = tempfile.mkdtemp()
                    paths = []
                    for uf in uploaded_files:
                        dest = os.path.join(tmp_dir, uf.name)
                        with open(dest, "wb") as f:
                            f.write(uf.read())
                        paths.append(dest)

                    embeddings = get_embeddings("huggingface")
                    st.session_state.vector_store = ingest_documents(paths, embeddings)
                    st.session_state.doc_names = [uf.name for uf in uploaded_files]
                    st.success(f"✅ Indexed {len(paths)} document(s)")
                except Exception as e:
                    st.error(f"RAG build failed: {e}")

        if st.session_state.doc_names:
            st.caption("**Indexed:**")
            for name in st.session_state.doc_names:
                st.caption(f"  • {name}")

        # ── Custom Persona ────────────────────────────────────────────────────
        st.divider()
        st.subheader("🎭 Custom Persona (optional)")
        st.session_state.custom_persona = st.text_area(
            "System prompt override",
            value=st.session_state.custom_persona,
            height=100,
            placeholder="e.g. You are a concise legal assistant…",
        )

        # ── Clear chat ────────────────────────────────────────────────────────
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── LLM call ─────────────────────────────────────────────────────────────────
def get_chat_response(system_prompt: str, messages: list) -> str:
    """Send message history + system prompt to the LLM and return the reply."""
    try:
        model = get_model(st.session_state.llm_provider)
        formatted = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted.append(HumanMessage(content=msg["content"]))
            else:
                formatted.append(AIMessage(content=msg["content"]))
        response = model.invoke(formatted)
        return response.content
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"⚠️ Error from LLM: {exc}"


# ── Chat page ─────────────────────────────────────────────────────────────────
def chat_page():
    st.title("🤖 NeoBot — Intelligent AI Assistant")

    # Feature badges
    badges = []
    if st.session_state.vector_store:
        badges.append('<span class="badge badge-rag">📚 RAG Active</span>')
    if st.session_state.use_web_search:
        badges.append('<span class="badge badge-web">🌐 Web Search On</span>')
    badges.append(
        f'<span class="badge badge-llm">🧠 {st.session_state.llm_provider.capitalize()}</span>'
    )
    badges.append(
        f'<span class="badge badge-llm">📝 {st.session_state.response_mode.capitalize()}</span>'
    )
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.caption("Ask me anything! Upload documents for RAG or let me search the web for fresh info.")
    st.divider()

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📎 Sources used"):
                    st.markdown(message["sources"])

    # Chat input
    if prompt := st.chat_input("Type your message here…"):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ── Build system prompt ───────────────────────────────────────────────
        rag_context = ""
        search_results = ""
        source_notes = []

        if st.session_state.vector_store:
            with st.spinner("🔍 Searching knowledge base…"):
                try:
                    rag_context = retrieve_context(prompt, st.session_state.vector_store)
                    if rag_context:
                        source_notes.append("📚 Retrieved context from uploaded documents.")
                except Exception as e:
                    logger.error("RAG retrieval error: %s", e)

        if st.session_state.use_web_search and should_search(prompt):
            with st.spinner("🌐 Searching the web…"):
                try:
                    search_results = web_search(prompt)
                    if search_results:
                        source_notes.append("🌐 Augmented with live web search results.")
                except Exception as e:
                    logger.error("Web search error: %s", e)

        system_prompt = build_system_prompt(
            rag_context=rag_context,
            search_results=search_results,
            response_mode=st.session_state.response_mode,
            custom_persona=st.session_state.custom_persona,
        )

        # ── Get and display response ──────────────────────────────────────────
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = get_chat_response(system_prompt, st.session_state.messages)
            st.markdown(response)
            if source_notes:
                with st.expander("📎 Sources used"):
                    st.markdown("\n\n".join(source_notes))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": "\n\n".join(source_notes) if source_notes else "",
            }
        )


# ── Instructions page ─────────────────────────────────────────────────────────
def instructions_page():
    st.title("📖 The Chatbot Blueprint — Setup Guide")
    st.markdown("""
## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🔑 API Keys

Set these as environment variables (or in Streamlit Cloud secrets):

| Variable | Required | Source |
|---|---|---|
| `GROQ_API_KEY` | For Groq models | [console.groq.com](https://console.groq.com/keys) |
| `OPENAI_API_KEY` | For OpenAI models | [platform.openai.com](https://platform.openai.com/api-keys) |
| `GOOGLE_API_KEY` | For Gemini models | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `TAVILY_API_KEY` | For web search | [tavily.com](https://tavily.com) |

## 🚀 Features

### RAG (Retrieval-Augmented Generation)
1. Upload PDF / TXT / DOCX / MD files in the sidebar
2. Click **Build Knowledge Base**
3. Ask questions — NeoBot retrieves relevant excerpts from your docs

### Live Web Search
- Toggle on in the sidebar
- NeoBot automatically decides when to search based on your query
- Results are cited in the *Sources used* expander

### Response Modes
- **Concise** — short, summarised answers (2-4 sentences)
- **Detailed** — thorough, structured explanations

### Custom Persona
- Override the default assistant persona with your own system prompt

## 🏃 Run Locally

```bash
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud

1. Push to a public GitHub repo
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Add your API keys as **Secrets** in the Streamlit dashboard
""")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    with st.sidebar:
        st.divider()
        page = st.radio("📍 Navigation", ["💬 Chat", "📖 Instructions"], index=0)

    if page == "📖 Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
