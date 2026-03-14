"""
service.py — Reusable local document ingest/query service.

Single-document per session, multiple sessions in memory.
Used by the local REST API and can also be reused by other interfaces.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import pandas as pd
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from nltk.stem import PorterStemmer

from config import WORK_DIR
from engine import build_engine, build_llm
from history import save_turn
from indexer import build_index
from ocr import auto_summarize, extract_document, make_zip
from prompts import (
    OVERVIEW_PROMPT_TEMPLATE,
    RELATIONSHIP_PROMPT_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    VISUAL_QA_PROMPT_TEMPLATE,
)

_STEMMER = PorterStemmer()


_OVERVIEW_RE = re.compile(
    r"\b("
    r"what\s+is\s+(this|the)\s+(document|doc|pdf|image|file)"
    r"|tell\s+me\s+about\s+(this|the)\s+(document|doc|pdf|image|file)"
    r"|what\s+is\s+(it|this)\s+about"
    r"|summari[sz]e\s+(this|the)\s+(document|doc|pdf|image|file)"
    r"|document\s+overview"
    r"|overview"
    r")\b",
    re.IGNORECASE,
)

_SYNTHESIS_RE = re.compile(
    r"\b("
    r"make|create|build|generate|show|display|give|list|compile|find|extract"
    r")\b.*\b("
    r"table|list|summary|overview|all|every|each|entries|records|data|owners|animals|diagnoses|diagnosis|treatments|phones|mobiles"
    r")\b"
    r"|\bhow many\b"
    r"|\bwhat (else|other|more)\b"
    r"|\bshow me (everything|all|the full|complete)\b",
    re.IGNORECASE,
)

_SYNTHESIS_WORDS = {
    "owner", "owners", "animal", "animals", "name", "names", "address", "addresses",
    "diagnosis", "diagnoses", "treatment", "treatments", "phone", "phones",
    "mobile", "mobiles", "table", "list", "entries", "records", "cases", "all",
}

_VISUAL_RE = re.compile(
    r"\b("
    r"what\s+does\s+(the\s+)?(image|photo|picture|figure|screenshot|page)\s+show"
    r"|describe\s+(the\s+)?(image|photo|picture|figure|screenshot|page)"
    r"|what\s+is\s+(shown|happening|visible|in\s+the\s+image|in\s+this\s+image)"
    r"|what\s+can\s+you\s+see"
    r"|what\s+does\s+it\s+look\s+like"
    r"|what\s+is\s+in\s+(the\s+)?(image|photo|picture|figure|screenshot)"
    r")\b",
    re.IGNORECASE,
)

_RELATIONSHIP_RE = re.compile(
    r"\b("
    r"link|linked|connect|connected|correspond|corresponding|match|matched|map|mapped"
    r"|join|joined|relate|related|relationship|specific|whose|which|who\s+had"
    r"|for\s+each|per\s+"
    r")\b",
    re.IGNORECASE,
)

_TABLE_FORMAT_RE = re.compile(
    r"\b(table|markdown(?:\s+table)?|rows?|columns?)\b",
    re.IGNORECASE,
)

_KEY_COL_HINTS = {
    "id", "key", "ref", "reference", "code", "serial", "sr", "case",
    "reg", "registration", "yearly", "monthly", "daily", "token",
    "no", "number", "num",
}
_SCHEMA_STOP_WORDS = {"of", "and", "the", "a", "an", "to", "by", "for"}


def _clean_node_text(text: str) -> str:
    return re.sub(
        r"\[(?:Doc:[^\]]+|Page \d+|TABLE ROW|VISUAL SUMMARY|HEADER|SECTION|FIGURE|NOTES|TABLE)\]",
        "",
        text or "",
    ).strip()


def _extract_page_numbers(question: str) -> list[int]:
    pages: set[int] = set()
    for match in re.finditer(r"\bpages?\s+([0-9,\sand\-]+)", question.lower()):
        for num in re.findall(r"\d+", match.group(1)):
            try:
                pages.add(int(num))
            except ValueError:
                continue
    return sorted(pages)


def _is_overview_query(question: str) -> bool:
    return bool(_OVERVIEW_RE.search(question.strip()))


def _is_synthesis_query(question: str, table_count: int) -> bool:
    if _is_overview_query(question):
        return False
    if _SYNTHESIS_RE.search(question):
        return True
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    return bool(len(words) <= 8 and words & _SYNTHESIS_WORDS and table_count > 1)


def _is_visual_query(question: str) -> bool:
    return bool(_VISUAL_RE.search(question.strip()))


def _is_relationship_query(question: str, tables: list[dict[str, Any]]) -> bool:
    if not tables:
        return False
    q_tokens = {_STEMMER.stem(token) for token in re.findall(r"[a-z0-9]+", question.lower())}
    families = _relevant_table_families(tables, q_tokens)
    if len(families) < 2:
        return False
    if _RELATIONSHIP_RE.search(question.strip()):
        return True
    matched_groups = sum(1 for family in families if family["matched_tokens"])
    return matched_groups >= 2


def _format_guidance(question: str) -> str:
    if _TABLE_FORMAT_RE.search(question):
        return (
            "Use a Markdown table if the answer is row- or column-oriented. "
            "Otherwise keep it concise."
        )
    return (
        "Answer in concise prose by default. Use a short bullet list if helpful. "
        "Do not use a Markdown table unless the user explicitly asks for one or "
        "the answer truly requires multiple rows and columns."
    )


def _apply_format_guidance(question: str) -> str:
    return f"{question}\n\nFormatting instruction: {_format_guidance(question)}"


def _page_context_text(page_data: dict[str, Any], text_limit: int = 2500) -> str:
    blocks = [f"[PAGE {page_data['page']}]"]
    visual_summary = (page_data.get("visual_summary") or "").strip()
    text = (page_data.get("text") or "").strip()
    if visual_summary:
        blocks.append(f"[VISUAL SUMMARY]\n{visual_summary}")
    if text:
        if len(text) > text_limit:
            text = text[:text_limit] + "..."
        blocks.append(f"[OCR TEXT]\n{text}")
    return "\n".join(blocks)


def _combined_page_text(page_data: dict[str, Any]) -> str:
    parts = []
    visual_summary = (page_data.get("visual_summary") or "").strip()
    text = (page_data.get("text") or "").strip()
    if visual_summary:
        parts.append(f"[VISUAL SUMMARY] {visual_summary}")
    if text:
        parts.append(text)
    return "\n".join(parts).strip()


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _column_tokens(cols: list[str]) -> set[str]:
    tokens: set[str] = set()
    for col in cols:
        for token in _normalize_col_name(col).split():
            if len(token) >= 3 and token not in _SCHEMA_STOP_WORDS:
                tokens.add(_STEMMER.stem(token))
    return tokens


def _table_families(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[tuple[str, ...], dict[str, Any]] = {}
    for table in tables:
        cols = [_normalize_col_name(col) for col in table.get("cols", [])]
        signature = tuple(cols)
        if signature not in families:
            families[signature] = {"cols": cols, "tables": [], "tokens": _column_tokens(cols)}
        families[signature]["tables"].append(table)
    return list(families.values())


def _tokens_related(a: str, b: str) -> bool:
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 5 and longer.startswith(shorter[:5])


def _relevant_table_families(tables: list[dict[str, Any]], q_tokens: set[str]) -> list[dict[str, Any]]:
    relevant = []
    for family in _table_families(tables):
        matched_tokens = {
            q_token
            for q_token in q_tokens
            if any(_tokens_related(q_token, family_token) for family_token in family["tokens"])
        }
        if matched_tokens:
            relevant.append({**family, "matched_tokens": matched_tokens})
    return relevant


def _candidate_key_cols(cols: list[str]) -> set[str]:
    keys = set()
    for col in cols:
        norm = _normalize_col_name(col)
        if any(hint in norm.split() for hint in _KEY_COL_HINTS):
            keys.add(norm)
    return keys


def _table_join_analysis(
    tables: list[dict[str, Any]],
    page_filter: list[int] | None = None,
    question: str | None = None,
) -> str:
    selected = [t for t in tables if not page_filter or t["page"] in page_filter]
    if not selected:
        return "Use only the directly provided table rows."

    q_tokens = {_STEMMER.stem(token) for token in re.findall(r"[a-z0-9]+", (question or "").lower())}
    families = _relevant_table_families(selected, q_tokens) if q_tokens else _table_families(selected)
    if len(families) <= 1:
        return "The question maps to one table schema only. Use direct row evidence and do not infer extra relationships."

    common_keys = set(_candidate_key_cols(families[0]["cols"]))
    for family in families[1:]:
        common_keys &= _candidate_key_cols(family["cols"])

    if common_keys:
        return (
            "Potential shared key columns detected across the relevant table groups: "
            + ", ".join(sorted(common_keys))
            + ". Join rows only when the actual values match exactly."
        )

    return (
        "No reliable shared key columns were detected across the relevant table groups. "
        "Do not map values between them unless the same identifier is visibly present in both tables."
    )


def _load_table_frame(table: dict[str, Any]) -> pd.DataFrame | None:
    try:
        return pd.read_csv(table["path"]).fillna("").astype(str)
    except Exception:
        return None


def _clean_table_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    weekday_tokens = {
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "new", "old",
    }
    for _, row in df.iterrows():
        cleaned: dict[str, str] = {}
        for col, value in zip(df.columns, row):
            text = str(value).strip()
            if not text or text.lower() in {"nan", '""'}:
                continue
            cleaned[str(col).strip()] = text
        if not cleaned:
            continue
        row_text = " ".join(cleaned.values())
        normalized_values = {_normalize_col_name(value) for value in cleaned.values()}
        if normalized_values and normalized_values <= weekday_tokens:
            continue
        # Purely numeric/date/id rows are weak anchors for layout-based joins and
        # often come from date stamps or scan artifacts instead of real records.
        if not any(ch.isalpha() for ch in row_text):
            continue
        if len(cleaned) <= 1:
            continue
        rows.append(cleaned)
    return rows


def _family_signature_from_cols(cols: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_col_name(col) for col in cols)


def _merge_row_dicts(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for key, value in left.items():
        merged[key] = value
    for key, value in right.items():
        out_key = key
        suffix = 2
        while out_key in merged:
            out_key = f"{key} ({suffix})"
            suffix += 1
        merged[out_key] = value
    return merged


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.72:
        return "medium"
    return "low"


def _ordered_relevant_families(tables: list[dict[str, Any]], q_tokens: set[str]) -> list[dict[str, Any]]:
    families = _relevant_table_families(tables, q_tokens) if q_tokens else _table_families(tables)
    ordered = []
    for family in families:
        earliest_page = min(table["page"] for table in family["tables"])
        ordered.append({**family, "signature": _family_signature_from_cols(family["cols"]), "earliest_page": earliest_page})
    return sorted(ordered, key=lambda family: (family["earliest_page"], len(family["cols"])))


def _direction_pairs(
    occurrences: list[dict[str, Any]],
    left_sig: tuple[str, ...],
    right_sig: tuple[str, ...],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_occurrences = [occ for occ in occurrences if occ["signature"] == left_sig]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, left in enumerate(left_occurrences):
        next_left_page = left_occurrences[index + 1]["page"] if index + 1 < len(left_occurrences) else float("inf")
        for occ in occurrences:
            if occ["page"] <= left["page"]:
                continue
            if occ["page"] >= next_left_page:
                break
            if occ["signature"] == right_sig and 1 <= occ["page"] - left["page"] <= 2:
                pairs.append((left, occ))
                break
    return pairs


def _build_inferred_join_context(
    tables: list[dict[str, Any]],
    question: str,
    page_filter: list[int] | None = None,
    char_limit: int = 18000,
) -> tuple[str, list[int], bool, list[dict[str, Any]]]:
    selected = [table for table in tables if not page_filter or table["page"] in page_filter]
    if len(selected) < 2:
        return "", [], False, []

    q_tokens = {_STEMMER.stem(token) for token in re.findall(r"[a-z0-9]+", question.lower())}
    families = _ordered_relevant_families(selected, q_tokens)
    if len(families) < 2:
        return "", [], False, []

    relevant_signatures = {family["signature"] for family in families}
    occurrences = []
    for table in sorted(selected, key=lambda item: (item["page"], item["path"])):
        signature = _family_signature_from_cols(table.get("cols", []))
        if signature not in relevant_signatures:
            continue
        frame = _load_table_frame(table)
        if frame is None:
            continue
        rows = _clean_table_rows(frame)
        if not rows:
            continue
        occurrences.append(
            {
                "page": table["page"],
                "signature": signature,
                "rows": rows,
                "table": table,
            }
        )

    blocks = []
    total = 0
    used_pages: list[int] = []
    truncated = False
    summaries: list[dict[str, Any]] = []

    for left_family, right_family in zip(families, families[1:]):
        left_sig = left_family["signature"]
        right_sig = right_family["signature"]
        pairs = _direction_pairs(occurrences, left_sig, right_sig)
        if not pairs:
            continue

        left_total = sum(1 for occ in occurrences if occ["signature"] == left_sig)
        right_total = sum(1 for occ in occurrences if occ["signature"] == right_sig)
        direction_strength = len(pairs) / max(1, min(left_total, right_total))

        for left_occ, right_occ in pairs:
            row_ratio = min(len(left_occ["rows"]), len(right_occ["rows"])) / max(len(left_occ["rows"]), len(right_occ["rows"]))
            if row_ratio < 0.8:
                continue

            page_gap = right_occ["page"] - left_occ["page"]
            gap_score = 1.0 if page_gap == 1 else 0.85
            matched_token_score = min(
                1.0,
                (len(left_family.get("matched_tokens", set())) + len(right_family.get("matched_tokens", set()))) / 4,
            )
            confidence = (
                0.45 * direction_strength
                + 0.35 * row_ratio
                + 0.15 * gap_score
                + 0.05 * matched_token_score
            )
            if confidence < 0.78:
                continue

            pair_count = min(len(left_occ["rows"]), len(right_occ["rows"]))
            merged_rows = [
                _merge_row_dicts(left_occ["rows"][idx], right_occ["rows"][idx])
                for idx in range(pair_count)
            ]
            if len(merged_rows) < 2:
                continue
            merged_df = pd.DataFrame(merged_rows).fillna("")

            block = (
                f"[INFERRED JOIN · confidence={_confidence_label(confidence)} · pages {left_occ['page']}->{right_occ['page']} "
                f"· basis=adjacent page pairing + row-order alignment]\n"
                "[LAYOUT-INFERRED CSV]\n"
                + merged_df.to_csv(index=False)
            )
            if blocks and total + len(block) > char_limit:
                truncated = True
                break

            blocks.append(block)
            total += len(block)
            for page_num in (left_occ["page"], right_occ["page"]):
                if page_num not in used_pages:
                    used_pages.append(page_num)
            summaries.append(
                {
                    "left_page": left_occ["page"],
                    "right_page": right_occ["page"],
                    "confidence": confidence,
                    "row_ratio": row_ratio,
                    "pair_count": pair_count,
                }
            )
        if truncated:
            break

    return "\n\n".join(blocks), used_pages, truncated, summaries


def _compose_join_analysis(
    tables: list[dict[str, Any]],
    question: str,
    page_filter: list[int] | None = None,
) -> tuple[str, str, list[int], bool]:
    base_analysis = _table_join_analysis(tables, page_filter=page_filter, question=question)
    inferred_ctx, inferred_pages, inferred_truncated, inferred_summaries = _build_inferred_join_context(
        tables,
        question=question,
        page_filter=page_filter,
    )
    if not inferred_summaries:
        return base_analysis, inferred_ctx, inferred_pages, inferred_truncated

    summary_lines = [
        "In addition, high-confidence inferred row-alignment candidates exist for:"
    ]
    for item in inferred_summaries:
        summary_lines.append(
            f"- pages {item['left_page']} -> {item['right_page']} "
            f"({_confidence_label(item['confidence'])} confidence, "
            f"row match {item['pair_count']} rows, ratio {item['row_ratio']:.2f})"
        )
    summary_lines.append(
        "These inferred joins are not exact-key joins. Use them only when clearly labeled as inferred."
    )
    return base_analysis + "\n" + "\n".join(summary_lines), inferred_ctx, inferred_pages, inferred_truncated


@dataclass
class DocumentSession:
    session_id: str
    doc_name: str
    file_path: str
    summary: str
    pages: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    combined_text: str
    zip_path: str | None
    engine: Any
    routing: dict[str, int] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class LocalDocQAService:
    def __init__(self) -> None:
        self.llm = build_llm()
        self._lock = RLock()
        self._sessions: dict[str, DocumentSession] = {}
        self._session_root = os.path.join(WORK_DIR, "api_sessions")
        os.makedirs(self._session_root, exist_ok=True)

    def _session_dir(self, session_id: str) -> str:
        path = os.path.join(self._session_root, session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _session_dir_path(self, session_id: str) -> str:
        return os.path.join(self._session_root, session_id)

    def _recent_history_for_prompt(self, session: DocumentSession, max_messages: int = 6) -> str:
        try:
            messages = session.engine._memory.get_all()
        except Exception:
            return "(no prior conversation)"
        lines = []
        for msg in messages[-max_messages:]:
            text = (msg.content or "").strip()
            if not text:
                continue
            if len(text) > 600:
                text = text[:600] + "..."
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            lines.append(f"{role}: {text}")
        return "\n".join(lines) if lines else "(no prior conversation)"

    def _remember_direct_turn(self, session: DocumentSession, question: str, answer: str) -> None:
        try:
            session.engine._memory.put(ChatMessage(content=question, role=MessageRole.USER))
            session.engine._memory.put(ChatMessage(content=answer, role=MessageRole.ASSISTANT))
        except Exception:
            pass

    def _complete_prompt(self, prompt: str) -> str:
        response = self.llm.complete(prompt)
        return str(response).strip()

    def _build_table_context(
        self, session: DocumentSession, page_filter: list[int] | None = None, char_limit: int = 28000
    ) -> tuple[str, list[int], bool]:
        blocks = []
        total = 0
        used_pages: list[int] = []
        truncated = False
        for table in session.tables:
            if page_filter and table["page"] not in page_filter:
                continue
            try:
                df = pd.read_csv(table["path"]).fillna("").astype(str)
            except Exception:
                continue
            block = (
                f"[STRUCTURED TABLE · Page {table['page']} · {table['rows']} rows × {len(table['cols'])} cols]\n"
                + df.to_string(index=False)
            )
            if blocks and total + len(block) > char_limit:
                truncated = True
                break
            blocks.append(block)
            total += len(block)
            if table["page"] not in used_pages:
                used_pages.append(table["page"])
        return "\n\n".join(blocks), used_pages, truncated

    def _build_overview_context(
        self, session: DocumentSession, page_filter: list[int] | None = None, char_limit: int = 18000
    ) -> tuple[str, list[int], bool]:
        blocks = []
        total = 0
        used_pages: list[int] = []
        truncated = False
        for page_data in session.pages:
            page_num = page_data["page"]
            if page_filter and page_num not in page_filter:
                continue
            block = _page_context_text(page_data)
            if block.strip() == f"[PAGE {page_num}]":
                continue
            if blocks and total + len(block) > char_limit:
                truncated = True
                break
            blocks.append(block)
            total += len(block)
            used_pages.append(page_num)
        return "\n\n".join(blocks), used_pages, truncated

    def _build_visual_context(
        self, session: DocumentSession, page_filter: list[int] | None = None, char_limit: int = 12000
    ) -> tuple[str, list[int], bool]:
        blocks = []
        total = 0
        used_pages: list[int] = []
        truncated = False
        for page_data in session.pages:
            page_num = page_data["page"]
            if page_filter and page_num not in page_filter:
                continue
            visual_summary = (page_data.get("visual_summary") or "").strip()
            text = (page_data.get("text") or "").strip()
            if not visual_summary and not text:
                continue
            block_parts = [f"[PAGE {page_num}]"]
            if visual_summary:
                block_parts.append(f"[VISUAL SUMMARY]\n{visual_summary}")
            if text and len(text) <= 500:
                block_parts.append(f"[OCR CLUES]\n{text}")
            block = "\n".join(block_parts)
            if blocks and total + len(block) > char_limit:
                truncated = True
                break
            blocks.append(block)
            total += len(block)
            used_pages.append(page_num)
        return "\n\n".join(blocks), used_pages, truncated

    def _serialize_source_nodes(self, source_nodes: list[Any]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for item in source_nodes or []:
            node = item.node
            serialized.append(
                {
                    "page": int(node.metadata.get("page_label", 0) or 0) or None,
                    "doc_name": node.metadata.get("doc_name"),
                    "score": float(item.score or 0.0),
                    "is_table_row": bool(node.metadata.get("is_table_row", False)),
                    "is_visual_summary": bool(node.metadata.get("is_visual_summary", False)),
                    "excerpt": _clean_node_text(node.text)[:500],
                }
            )
        return serialized

    def _ingest_stored_file(self, stored_path: str, doc_name: str, session_id: str) -> dict[str, Any]:
        extracted = extract_document(stored_path, doc_name)
        pages = extracted["pages"]
        tables = extracted["tables"]
        routing = extracted.get("routing", {})

        combined_text = "\n\n".join(
            _combined_page_text(page_data)
            for page_data in pages
            if _combined_page_text(page_data)
        ).strip()
        if not combined_text:
            raise ValueError("No readable text or visual description could be extracted from the uploaded document.")

        summary = auto_summarize(combined_text)
        index = build_index(pages, tables, doc_name)
        engine = build_engine(index, summary, self.llm)
        zip_path = make_zip(tables, doc_name)

        session = DocumentSession(
            session_id=session_id,
            doc_name=doc_name,
            file_path=stored_path,
            summary=summary,
            pages=pages,
            tables=tables,
            combined_text=combined_text,
            zip_path=zip_path,
            engine=engine,
            routing=routing,
        )

        with self._lock:
            self._sessions[session_id] = session

        return self.get_session_info(session_id)

    def ingest_path(self, path: str, original_name: str | None = None) -> dict[str, Any]:
        source_path = os.path.abspath(path)
        doc_name = original_name or os.path.basename(source_path)
        session_id = uuid4().hex[:12]
        session_dir = self._session_dir(session_id)

        ext = Path(doc_name).suffix.lower()
        stored_path = os.path.join(session_dir, f"uploaded{ext or ''}")
        shutil.copy2(source_path, stored_path)
        return self._ingest_stored_file(stored_path, doc_name, session_id)

    def ingest_bytes(self, filename: str, data: bytes) -> dict[str, Any]:
        session_id = uuid4().hex[:12]
        session_dir = self._session_dir(session_id)
        ext = Path(filename).suffix.lower()
        stored_path = os.path.join(session_dir, f"uploaded{ext or ''}")
        with open(stored_path, "wb") as f:
            f.write(data)
        return self._ingest_stored_file(stored_path, filename, session_id)

    def get_session(self, session_id: str) -> DocumentSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return {
            "session_id": session.session_id,
            "doc_name": session.doc_name,
            "summary": session.summary,
            "page_count": len(session.pages),
            "table_count": len(session.tables),
            "table_zip_path": session.zip_path,
            "routing": session.routing,
            "created_at": session.created_at,
        }

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        try:
            shutil.rmtree(self._session_dir_path(session_id), ignore_errors=True)
        except Exception:
            pass
        return True

    def query_session(self, session_id: str, question: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        question = question.strip()
        page_filter = _extract_page_numbers(question)

        if _is_visual_query(question):
            context, used_pages, truncated = self._build_visual_context(session, page_filter or None)
            if context:
                prompt = VISUAL_QA_PROMPT_TEMPLATE.format(
                    doc_summary=session.summary,
                    history=self._recent_history_for_prompt(session),
                    context=context,
                    question=question,
                    format_guidance=_format_guidance(question),
                )
                answer = self._complete_prompt(prompt)
                self._remember_direct_turn(session, question, answer)
                save_turn(session.doc_name, question, answer, "document-visual")
                return {
                    "session_id": session_id,
                    "mode": "visual",
                    "question": question,
                    "answer": answer,
                    "doc_name": session.doc_name,
                    "summary": session.summary,
                    "source_pages": used_pages,
                    "truncated": truncated,
                    "sources": [],
                }

        if _is_overview_query(question):
            context, used_pages, truncated = self._build_overview_context(session, page_filter or None)
            if not context:
                context = session.combined_text[:6000]
                used_pages = page_filter or [p["page"] for p in session.pages[:3]]
                truncated = len(session.combined_text) > len(context)

            prompt = OVERVIEW_PROMPT_TEMPLATE.format(
                doc_summary=session.summary,
                history=self._recent_history_for_prompt(session),
                context=context or "(no document text available)",
                question=question,
                format_guidance=_format_guidance(question),
            )
            answer = self._complete_prompt(prompt)
            self._remember_direct_turn(session, question, answer)
            save_turn(session.doc_name, question, answer, "document-overview")
            return {
                "session_id": session_id,
                "mode": "overview",
                "question": question,
                "answer": answer,
                "doc_name": session.doc_name,
                "summary": session.summary,
                "source_pages": used_pages,
                "truncated": truncated,
                "sources": [],
            }

        if _is_relationship_query(question, session.tables):
            context, used_pages, truncated = self._build_table_context(session, page_filter or None)
            if context:
                join_analysis, inferred_ctx, inferred_pages, inferred_truncated = _compose_join_analysis(
                    session.tables,
                    question=question,
                    page_filter=page_filter or None,
                )
                if inferred_ctx:
                    context = inferred_ctx + "\n\n[RAW TABLES]\n" + context
                    for page_num in inferred_pages:
                        if page_num not in used_pages:
                            used_pages.append(page_num)
                    truncated = truncated or inferred_truncated
                prompt = RELATIONSHIP_PROMPT_TEMPLATE.format(
                    history=self._recent_history_for_prompt(session),
                    question=question,
                    format_guidance=_format_guidance(question),
                    join_analysis=join_analysis,
                    context=context,
                )
                answer = self._complete_prompt(prompt)
                self._remember_direct_turn(session, question, answer)
                save_turn(session.doc_name, question, answer, "document-relationship")
                return {
                    "session_id": session_id,
                    "mode": "relationship",
                    "question": question,
                    "answer": answer,
                    "doc_name": session.doc_name,
                    "summary": session.summary,
                    "source_pages": used_pages,
                    "truncated": truncated,
                    "sources": [
                        {
                            "page": table["page"],
                            "rows": table["rows"],
                            "cols": table["cols"],
                            "path": table["path"],
                        }
                        for table in session.tables
                        if not used_pages or table["page"] in used_pages
                    ],
                }

        if _is_synthesis_query(question, len(session.tables)):
            context, used_pages, truncated = self._build_table_context(session, page_filter or None)
            if context:
                join_analysis, inferred_ctx, inferred_pages, inferred_truncated = _compose_join_analysis(
                    session.tables,
                    question=question,
                    page_filter=page_filter or None,
                )
                if inferred_ctx:
                    context = inferred_ctx + "\n\n[RAW TABLES]\n" + context
                    for page_num in inferred_pages:
                        if page_num not in used_pages:
                            used_pages.append(page_num)
                    truncated = truncated or inferred_truncated
                prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                    history=self._recent_history_for_prompt(session),
                    question=question,
                    format_guidance=_format_guidance(question),
                    join_analysis=join_analysis,
                    context=context,
                )
                answer = self._complete_prompt(prompt)
                self._remember_direct_turn(session, question, answer)
                save_turn(session.doc_name, question, answer, "document-synthesis")
                return {
                    "session_id": session_id,
                    "mode": "synthesis",
                    "question": question,
                    "answer": answer,
                    "doc_name": session.doc_name,
                    "summary": session.summary,
                    "source_pages": used_pages,
                    "truncated": truncated,
                    "sources": [
                        {
                            "page": table["page"],
                            "rows": table["rows"],
                            "cols": table["cols"],
                            "path": table["path"],
                        }
                        for table in session.tables
                        if not used_pages or table["page"] in used_pages
                    ],
                }

        response = session.engine.chat(_apply_format_guidance(question))
        save_turn(session.doc_name, question, response.response, "document")
        sources = self._serialize_source_nodes(response.source_nodes)
        pages = sorted({src["page"] for src in sources if src["page"] is not None})
        return {
            "session_id": session_id,
            "mode": "document",
            "question": question,
            "answer": response.response,
            "doc_name": session.doc_name,
            "summary": session.summary,
            "source_pages": pages,
            "truncated": False,
            "sources": sources,
        }


SERVICE = LocalDocQAService()
