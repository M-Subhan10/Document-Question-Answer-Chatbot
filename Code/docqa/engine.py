"""
engine.py — Build and manage the LlamaIndex chat engine.

Uses:
  CondensePlusContextChatEngine — the right choice because:
    1. Condense step: rewrites follow-up questions into standalone queries
       ("what about the second one?" → "what is the address of the second
        animal owner in the register?") — no hand-rolled query rewriting needed
    2. Context step: retrieves relevant nodes via the VectorStoreIndex retriever
    3. Response step: LLM answers using retrieved context + chat history
    4. Memory: ChatSummaryMemoryBuffer auto-summarises old messages when
       approaching the token limit — much smarter than a fixed sliding window

Public API:
  build_engine(index, doc_summary, llm)  -> CondensePlusContextChatEngine
  build_llm()                            -> LLM (OpenAI or Gemini)
"""

from llama_index.core import Settings
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from config import (
    BACKEND, OPENAI_KEY, GEMINI_KEY,
    QA_MODEL, GEMINI_MODEL, QA_TOKENS,
    MEMORY_TOKEN_LIMIT, TOP_K,
)
from prompts import SYSTEM_PROMPT_TEMPLATE


def build_llm():
    """Construct the LLM for the selected backend and set it globally."""
    if BACKEND == "openai":
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(
            model=QA_MODEL,
            api_key=OPENAI_KEY,
            temperature=0.1,
            max_tokens=QA_TOKENS,
        )
    else:
        from llama_index.llms.gemini import Gemini
        llm = Gemini(
            model=GEMINI_MODEL,
            api_key=GEMINI_KEY,
            temperature=0.1,
            max_tokens=QA_TOKENS,
        )
    Settings.llm = llm
    print(f"LLM ready: {BACKEND} ({QA_MODEL if BACKEND == 'openai' else GEMINI_MODEL})")
    return llm


def build_engine(index, doc_summary: str, llm) -> CondensePlusContextChatEngine:
    """
    Build a fresh CondensePlusContextChatEngine for a newly loaded document.

    Called every time a new document is uploaded.
    The old engine is discarded — each document gets a clean conversation.

    Memory: ChatSummaryMemoryBuffer with token_limit=3000
      - Keeps the most recent messages in full
      - Summarises older messages when approaching the limit
      - Better than ChatMemoryBuffer which just drops old messages
      - Handles 20+ turns gracefully before summarisation kicks in
    """
    memory = ChatSummaryMemoryBuffer.from_defaults(
        token_limit=MEMORY_TOKEN_LIMIT,
        llm=llm,
    )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(doc_summary=doc_summary)

    retriever = index.as_retriever(similarity_top_k=TOP_K)

    engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=llm,
        memory=memory,
        system_prompt=system_prompt,
        verbose=False,
    )
    print("Chat engine ready (CondensePlusContext + ChatSummaryMemoryBuffer)")
    return engine
