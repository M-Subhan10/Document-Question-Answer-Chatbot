from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .types import ConversationMemory, DocumentRecord


@dataclass
class SessionState:
    docs: list[DocumentRecord] = field(default_factory=list)
    chunks: list = field(default_factory=list)
    vectors: np.ndarray | None = None
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    zip_path: str | None = None

    def reset_for_documents(self) -> None:
        self.memory = ConversationMemory()

    @property
    def has_docs(self) -> bool:
        return bool(self.docs and self.chunks and self.vectors is not None)


def load_history(settings: Settings) -> list[dict]:
    try:
        if settings.history_file.exists():
            return json.loads(settings.history_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def append_history(settings: Settings, doc_names: list[str], question: str, answer: str) -> None:
    sessions = load_history(settings)
    today = dt.date.today().isoformat()
    key = f"{' + '.join(doc_names) or '(no document)'}::{today}"
    session = next((item for item in sessions if item.get("key") == key), None)
    if session is None:
        session = {"key": key, "date": today, "docs": doc_names, "turns": []}
        sessions.append(session)
    session["turns"].append(
        {
            "time": dt.datetime.now().strftime("%H:%M"),
            "q": question,
            "a": answer,
        }
    )
    settings.history_file.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")
