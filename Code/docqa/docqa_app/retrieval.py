from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable

from .config import Settings
from .types import ChunkRecord, DocumentRecord, RetrievalScope, SearchHit

if TYPE_CHECKING:
    import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")
PAGE_RE = re.compile(r"\b(?:page|pages|pg|p\.)\s*([0-9,\sand\-]+)", re.IGNORECASE)
FOLLOW_UP_PHRASES = (
    "what about",
    "and what about",
    "this document",
    "that document",
    "this file",
    "that file",
    "this paper",
    "that paper",
    "same document",
    "same file",
)
ORDINAL_DOC_PHRASES = {
    1: ("first document", "first pdf", "first file", "first paper", "document 1", "pdf 1", "file 1", "paper 1"),
    2: ("second document", "second pdf", "second file", "second paper", "document 2", "pdf 2", "file 2", "paper 2"),
    3: ("third document", "third pdf", "third file", "third paper", "document 3", "pdf 3", "file 3", "paper 3"),
}


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def page_refs_from_question(question: str) -> list[int]:
    refs: list[int] = []
    for match in PAGE_RE.finditer(question):
        refs.extend(int(n) for n in re.findall(r"\d+", match.group(1)))
    return sorted(set(n for n in refs if n > 0))


def doc_refs_from_question(question: str, docs: list[DocumentRecord]) -> list[str]:
    lowered = question.lower()
    if any(phrase in lowered for phrase in ("all documents", "all docs", "all files", "both documents", "both files", "both papers")):
        return [doc.doc_id for doc in docs]

    matches: list[str] = []
    q_tokens = tokenize(lowered)
    for index, doc in enumerate(docs, start=1):
        if index in ORDINAL_DOC_PHRASES and any(phrase in lowered for phrase in ORDINAL_DOC_PHRASES[index]):
            matches.append(doc.doc_id)
            continue
        stem_tokens = tokenize(doc.name.replace("_", " ").replace("-", " "))
        if not stem_tokens:
            continue
        overlap = stem_tokens & q_tokens
        if overlap:
            matches.append(doc.doc_id)
    if "last document" in lowered or "latest document" in lowered or "last file" in lowered:
        matches.append(docs[-1].doc_id)
    return unique_in_order(matches)


def is_follow_up_query(question: str) -> bool:
    lowered = question.lower().strip()
    q_tokens = tokenize(lowered)
    pronouns = {"it", "its", "they", "them", "this", "that", "those", "these", "he", "she", "his", "her"}
    return (
        any(phrase in lowered for phrase in FOLLOW_UP_PHRASES)
        or bool(q_tokens & pronouns)
        or bool(page_refs_from_question(question))
    )


def is_overview_query(question: str) -> bool:
    q = question.lower().strip()
    overview_patterns = (
        "what is this",
        "what is this about",
        "what is the document about",
        "what is this document about",
        "what did i upload",
        "summarize the document",
        "summary of the document",
        "summarise the document",
        "summarize this paper",
        "what are these documents about",
    )
    return any(pattern in q for pattern in overview_patterns)


def is_external_query(question: str) -> bool:
    q = question.lower().strip()
    return q.startswith("/web ") or "outside the document" in q or "not about the document" in q


def split_page_chunks(text: str, settings: Settings) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= settings.chunk_words:
            chunks.append(para)
            continue
        start = 0
        while start < len(words):
            end = min(start + settings.chunk_words, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += settings.chunk_words - settings.overlap_words
    return chunks or [text]


def build_chunks(docs: Iterable[DocumentRecord], settings: Settings) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for doc in docs:
        if doc.summary:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}:summary",
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    page_num=0,
                    kind="doc-summary",
                    text=doc.summary,
                    token_set=tokenize(doc.summary),
                )
            )
        for page in doc.pages:
            page_summary = page.summary or page.text[:420]
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}:page:{page.page_num}:summary",
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    page_num=page.page_num,
                    kind="page-summary",
                    text=page_summary,
                    token_set=tokenize(page_summary),
                )
            )
            for idx, chunk_text in enumerate(split_page_chunks(page.text, settings), start=1):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc.doc_id}:page:{page.page_num}:text:{idx}",
                        doc_id=doc.doc_id,
                        doc_name=doc.name,
                        page_num=page.page_num,
                        kind="page-text",
                        text=chunk_text,
                        token_set=tokenize(chunk_text),
                    )
                )
            for table in page.tables:
                try:
                    import pandas as pd

                    df = pd.read_csv(table.path)
                    for idx, (_, row) in enumerate(df.iterrows(), start=1):
                        parts = [
                            f"{column}: {value}"
                            for column, value in zip(df.columns, row)
                            if str(value).strip() and str(value).strip().lower() not in {"nan", ""}
                        ]
                        if not parts:
                            continue
                        text = " | ".join(parts)
                        chunks.append(
                            ChunkRecord(
                                chunk_id=f"{doc.doc_id}:page:{page.page_num}:table:{idx}",
                                doc_id=doc.doc_id,
                                doc_name=doc.name,
                                page_num=page.page_num,
                                kind="table-row",
                                text=text,
                                token_set=tokenize(text),
                            )
                        )
                except Exception:
                    continue
    return chunks


class HybridRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = None
        self.prefix = "Represent this sentence for searching relevant passages: "

    def _ensure_embedder(self):
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.settings.embed_model)
        return self.embedder

    def build_vectors(self, chunks: list[ChunkRecord]) -> np.ndarray:
        texts = [self.prefix + chunk.text for chunk in chunks]
        vectors = self._ensure_embedder().encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        return vectors

    def scope_for(self, question: str, docs: list[DocumentRecord], memory) -> RetrievalScope:
        doc_ids = doc_refs_from_question(question, docs)
        if not doc_ids and memory.active_doc_ids and is_follow_up_query(question):
            doc_ids = list(memory.active_doc_ids)
        return RetrievalScope(
            doc_ids=unique_in_order(doc_ids),
            page_refs=page_refs_from_question(question),
            overview_query=is_overview_query(question),
            external_query=is_external_query(question),
        )

    def search(
        self,
        question: str,
        docs: list[DocumentRecord],
        chunks: list[ChunkRecord],
        vectors: np.ndarray,
        memory,
    ) -> tuple[RetrievalScope, list[SearchHit]]:
        scope = self.scope_for(question, docs, memory)
        if not chunks:
            return scope, []

        allowed_doc_ids = set(scope.doc_ids) if scope.doc_ids else {doc.doc_id for doc in docs}
        allowed_pages = set(scope.page_refs)
        query_tokens = tokenize(question)
        query_vec = self._ensure_embedder().encode(
            [self.prefix + question],
            normalize_embeddings=True,
        ).astype("float32")[0]
        dense_scores = vectors @ query_vec
        final_hits: list[SearchHit] = []

        for idx, chunk in enumerate(chunks):
            if chunk.doc_id not in allowed_doc_ids:
                continue
            if allowed_pages and chunk.page_num not in {0, *allowed_pages}:
                continue

            dense = float(dense_scores[idx])
            lexical = len(query_tokens & chunk.token_set) / max(1, len(query_tokens))
            bonus = 0.0
            if scope.overview_query and chunk.kind == "doc-summary":
                bonus += 0.25
            if chunk.kind == "table-row" and {"table", "row", "list", "all", "count"} & query_tokens:
                bonus += 0.08
            if chunk.page_num in memory.active_pages:
                bonus += 0.05
            if any(abs(chunk.page_num - page_num) == 1 for page_num in memory.active_pages if chunk.page_num):
                bonus += 0.02
            if chunk.doc_id in memory.active_doc_ids:
                bonus += 0.04
            if allowed_pages and chunk.page_num in allowed_pages:
                bonus += 0.12
            if scope.overview_query and chunk.kind == "page-summary":
                bonus += 0.05
            score = dense * 0.72 + lexical * 0.20 + bonus
            final_hits.append(SearchHit(chunk=chunk, score=score))

        final_hits.sort(key=lambda hit: hit.score, reverse=True)
        trimmed = final_hits[: self.settings.retrieval_k]
        expanded = self._expand_neighbor_pages(trimmed, chunks, memory)
        return scope, expanded[: self.settings.retrieval_k + 4]

    def _expand_neighbor_pages(self, hits: list[SearchHit], chunks: list[ChunkRecord], memory) -> list[SearchHit]:
        if not hits:
            return []
        boosted: dict[str, SearchHit] = {hit.chunk.chunk_id: hit for hit in hits}
        focus_pages: defaultdict[str, set[int]] = defaultdict(set)
        for hit in hits[:4]:
            if hit.chunk.page_num:
                focus_pages[hit.chunk.doc_id].add(hit.chunk.page_num)
        for chunk in chunks:
            if chunk.kind != "page-summary":
                continue
            if chunk.doc_id not in focus_pages:
                continue
            if any(abs(chunk.page_num - page_num) == 1 for page_num in focus_pages[chunk.doc_id]):
                boosted.setdefault(chunk.chunk_id, SearchHit(chunk=chunk, score=0.18))
        return sorted(boosted.values(), key=lambda hit: hit.score, reverse=True)
