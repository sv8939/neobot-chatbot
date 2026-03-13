"""
utils/rag_utils.py — Document ingestion and retrieval helpers for RAG.

Workflow
--------
1. User uploads one or more documents (PDF / TXT / DOCX / MD).
2. `ingest_documents()` loads, splits, embeds, and stores them in a FAISS index.
3. `retrieve_context()` fetches the top-K most relevant chunks for a query.
4. The caller prepends the retrieved context to the system prompt before
   sending messages to the LLM.
"""

import sys, os, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS

logger = logging.getLogger(__name__)


def _load_file(file_path: str):
    """Load a single file and return a list of LangChain Document objects."""
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredWordDocumentLoader,
    )

    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in (".txt", ".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext in (".docx", ".doc"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader.load()
    except Exception as exc:
        logger.error("Error loading %s: %s", file_path, exc)
        raise


def ingest_documents(file_paths: List[str], embeddings):
    """
    Ingest a list of documents and return a FAISS VectorStore.

    Parameters
    ----------
    file_paths : list[str]
        Absolute paths to uploaded documents.
    embeddings : Embeddings
        An initialised LangChain Embeddings object.

    Returns
    -------
    FAISS vector store ready for similarity search.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    all_docs = []
    for path in file_paths:
        try:
            docs = _load_file(path)
            all_docs.extend(docs)
            logger.info("Loaded %d pages/sections from %s", len(docs), path)
        except Exception as exc:
            logger.warning("Skipping %s due to error: %s", path, exc)

    if not all_docs:
        raise ValueError("No documents could be loaded. Check file formats.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    logger.info("Split into %d chunks", len(chunks))

    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS index built successfully.")
        return vector_store
    except Exception as exc:
        logger.error("Failed to build FAISS index: %s", exc)
        raise


def retrieve_context(query: str, vector_store, k: int = TOP_K_RESULTS) -> str:
    """
    Retrieve the top-K relevant chunks for a query and return them as a
    single context string to inject into the system prompt.

    Parameters
    ----------
    query : str
        The user's question.
    vector_store : FAISS
        A previously built FAISS vector store.
    k : int
        Number of chunks to retrieve.

    Returns
    -------
    str  — formatted context block, or empty string if nothing found.
    """
    try:
        results = vector_store.similarity_search(query, k=k)
        if not results:
            return ""
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Document")
            context_parts.append(f"[Excerpt {i} from {os.path.basename(source)}]\n{doc.page_content}")
        return "\n\n".join(context_parts)
    except Exception as exc:
        logger.error("Error during retrieval: %s", exc)
        return ""
