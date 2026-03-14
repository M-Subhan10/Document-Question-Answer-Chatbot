from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


FileKind = Literal["pdf", "image"]
PageSource = Literal["digital-text", "ocr-primary", "ocr-backup"]
ChunkKind = Literal["doc-summary", "page-summary", "page-text", "table-row"]


@dataclass
class TableArtifact:
    doc_id: str
    doc_name: str
    page_num: int
    path: Path
    rows: int
    cols: list[str]


@dataclass
class PageRecord:
    doc_id: str
    doc_name: str
    page_num: int
    text: str
    source: PageSource
    confidence: float
    warnings: list[str] = field(default_factory=list)
    summary: str = ""
    tables: list[TableArtifact] = field(default_factory=list)


@dataclass
class DocumentRecord:
    doc_id: str
    name: str
    kind: FileKind
    path: Path
    page_count: int
    pages: list[PageRecord]
    summary: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_num: int
    kind: ChunkKind
    text: str
    token_set: set[str]


@dataclass
class SearchHit:
    chunk: ChunkRecord
    score: float


@dataclass
class TurnMemory:
    question: str
    answer: str
    doc_ids: list[str]
    pages: list[int]


@dataclass
class ConversationMemory:
    rolling_summary: str = ""
    recent_turns: list[TurnMemory] = field(default_factory=list)
    active_doc_ids: list[str] = field(default_factory=list)
    active_pages: list[int] = field(default_factory=list)
    user_turns: int = 0
    notice: str = ""


@dataclass
class RetrievalScope:
    doc_ids: list[str] = field(default_factory=list)
    page_refs: list[int] = field(default_factory=list)
    overview_query: bool = False
    external_query: bool = False
