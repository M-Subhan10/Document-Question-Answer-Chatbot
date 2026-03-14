from __future__ import annotations

from dataclasses import asdict

from .config import Settings
from .prompts import load_prompt
from .types import ConversationMemory, SearchHit, TurnMemory


def format_recent_turns(memory: ConversationMemory) -> str:
    if not memory.recent_turns:
        return "(none)"
    lines = []
    for turn in memory.recent_turns:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")
    return "\n".join(lines)


def update_memory(
    memory: ConversationMemory,
    question: str,
    answer: str,
    hits: list[SearchHit],
    llm_complete,
    settings: Settings,
) -> tuple[ConversationMemory, bool]:
    doc_ids = []
    pages = []
    for hit in hits[:6]:
        if hit.chunk.doc_id not in doc_ids:
            doc_ids.append(hit.chunk.doc_id)
        if hit.chunk.page_num and hit.chunk.page_num not in pages:
            pages.append(hit.chunk.page_num)

    memory.user_turns += 1
    memory.active_doc_ids = doc_ids or memory.active_doc_ids
    memory.active_pages = pages or memory.active_pages
    memory.recent_turns.append(
        TurnMemory(question=question, answer=answer, doc_ids=doc_ids, pages=pages)
    )

    if len(memory.recent_turns) > 6:
        old_turns = memory.recent_turns[:-4]
        prompt = load_prompt("memory_rollup.txt").format(
            existing_summary=memory.rolling_summary or "(none)",
            recent_turns="\n".join(
                f"User: {turn.question}\nAssistant: {turn.answer}" for turn in old_turns
            ),
        )
        try:
            memory.rolling_summary = llm_complete(prompt).strip() or memory.rolling_summary
        except Exception:
            pass
        memory.recent_turns = memory.recent_turns[-4:]

    reset_visible_chat = False
    if memory.user_turns >= settings.rollover_after_turns:
        memory.notice = (
            "Chat context was archived to keep answers precise. "
            "Uploaded documents remain loaded."
        )
        prompt = load_prompt("memory_rollup.txt").format(
            existing_summary=memory.rolling_summary or "(none)",
            recent_turns=format_recent_turns(memory),
        )
        try:
            memory.rolling_summary = llm_complete(prompt).strip() or memory.rolling_summary
        except Exception:
            pass
        memory.recent_turns = []
        memory.user_turns = 0
        reset_visible_chat = True
    elif memory.user_turns >= settings.warn_after_turns:
        memory.notice = (
            f"Chat memory is getting long. After {settings.rollover_after_turns} user turns "
            "the visible chat will refresh, while document context stays loaded."
        )
    else:
        memory.notice = ""

    return memory, reset_visible_chat
