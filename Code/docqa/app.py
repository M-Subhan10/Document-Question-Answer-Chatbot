"""
app.py — Gradio UI for docqa_v3.

Thin orchestration layer. All real logic lives in the other modules.

Run:
  OPENAI_API_KEY='sk-...'  python app.py
  GEMINI_API_KEY='AIza...' python app.py

Architecture:
  Upload → ocr.py → indexer.py → engine.py → chat engine stored in STATE
  Ask    → engine.stream_chat(question) → stream tokens to Gradio chatbot
         → response.source_nodes → renderer.py → source panel HTML
"""

import os, re, time, traceback
from pathlib import Path
from datetime import datetime
from html import escape as _esc
import gradio as gr
import pandas as pd
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from nltk.stem import PorterStemmer

# ── Ensure Poppler is in PATH for PDF Processing ──────────────────────────────
poppler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "poppler", "poppler-24.02.0", "Library", "bin"))
if poppler_path not in os.environ["PATH"]:
    os.environ["PATH"] = poppler_path + os.pathsep + os.environ.get("PATH", "")

# ── Our modules ───────────────────────────────────────────────────────────────
from config import (
    BACKEND_LABEL, BADGE_COLOR, MSG_LIMIT, MSG_WARN_AT,
    GRADIO_SHARE, OPEN_BROWSER, SERVER_HOST, SERVER_PORT, WORK_DIR,
)
from ocr       import stream_document_extraction, make_zip, auto_summarize, transcribe_audio
from indexer   import build_index
from engine    import build_llm, build_engine
from history   import save_turn, render_history_html
from renderer  import (
    render_log, render_sources, TABLE_CSS,
    render_general_badge, render_chitchat_badge,
    render_context_cleared_badge, render_warn_badge,
    render_full_table_sources, render_overview_sources, render_visual_sources,
)
from prompts import (
    OVERVIEW_PROMPT_TEMPLATE,
    RELATIONSHIP_PROMPT_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    VISUAL_QA_PROMPT_TEMPLATE,
)

# ── LLM: build once at startup ────────────────────────────────────────────────
LLM = build_llm()
_STEMMER = PorterStemmer()

# ── Global state for the currently loaded document ───────────────────────────
# Single document only (no multi-doc for now).
STATE: dict = {
    "engine": None,    # CondensePlusContextChatEngine
    "tables": [],      # list of table metadata dicts from ocr.py
    "name":   "",      # original filename
    "summary": "",     # auto-generated one-sentence description
    "zip":    None,    # path to tables zip, or None
    "pages":  [],      # list of {'text': str, 'page': int}
    "text":   "",      # combined document text
}


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


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


def _is_synthesis_query(question: str) -> bool:
    if _is_overview_query(question):
        return False
    if _SYNTHESIS_RE.search(question):
        return True
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    return bool(len(words) <= 8 and words & _SYNTHESIS_WORDS and len(STATE.get("tables", [])) > 1)


def _is_visual_query(question: str) -> bool:
    return bool(_VISUAL_RE.search(question.strip()))


def _is_relationship_query(question: str) -> bool:
    if not STATE.get("tables"):
        return False
    q_tokens = {_STEMMER.stem(token) for token in re.findall(r"[a-z0-9]+", question.lower())}
    families = _relevant_table_families(STATE["tables"], q_tokens)
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


def _recent_history_for_prompt(engine, max_messages: int = 6) -> str:
    if engine is None:
        return ""
    try:
        messages = engine._memory.get_all()
    except Exception:
        return ""
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


def _remember_direct_turn(engine, question: str, answer: str) -> None:
    if engine is None:
        return
    try:
        engine._memory.put(ChatMessage(content=question, role=MessageRole.USER))
        engine._memory.put(ChatMessage(content=answer, role=MessageRole.ASSISTANT))
    except Exception:
        pass


def _page_context_text(page_data: dict, text_limit: int = 2500) -> str:
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


def _combined_page_text(page_data: dict) -> str:
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


def _table_families(tables: list[dict]) -> list[dict]:
    families: dict[tuple[str, ...], dict] = {}
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


def _relevant_table_families(tables: list[dict], q_tokens: set[str]) -> list[dict]:
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


def _table_join_analysis(page_filter: list[int] | None = None, question: str | None = None) -> str:
    selected = [t for t in STATE.get("tables", []) if not page_filter or t["page"] in page_filter]
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


def _build_table_context(page_filter: list[int] | None = None, char_limit: int = 28000):
    blocks = []
    total = 0
    used_pages: list[int] = []
    truncated = False
    for t in STATE.get("tables", []):
        if page_filter and t["page"] not in page_filter:
            continue
        try:
            df = pd.read_csv(t["path"]).fillna("").astype(str)
        except Exception:
            continue
        block = (
            f"[STRUCTURED TABLE · Page {t['page']} · {t['rows']} rows × {len(t['cols'])} cols]\n"
            + df.to_string(index=False)
        )
        if blocks and total + len(block) > char_limit:
            truncated = True
            break
        blocks.append(block)
        total += len(block)
        if t["page"] not in used_pages:
            used_pages.append(t["page"])
    return "\n\n".join(blocks), used_pages, truncated


def _load_table_frame(table: dict) -> pd.DataFrame | None:
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


def _ordered_relevant_families(selected: list[dict], q_tokens: set[str]) -> list[dict]:
    families = _relevant_table_families(selected, q_tokens) if q_tokens else _table_families(selected)
    ordered = []
    for family in families:
        earliest_page = min(table["page"] for table in family["tables"])
        ordered.append({**family, "signature": _family_signature_from_cols(family["cols"]), "earliest_page": earliest_page})
    return sorted(ordered, key=lambda family: (family["earliest_page"], len(family["cols"])))


def _direction_pairs(occurrences: list[dict], left_sig: tuple[str, ...], right_sig: tuple[str, ...]) -> list[tuple[dict, dict]]:
    left_occurrences = [occ for occ in occurrences if occ["signature"] == left_sig]
    pairs: list[tuple[dict, dict]] = []
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


def _build_inferred_join_context(question: str, page_filter: list[int] | None = None, char_limit: int = 18000):
    selected = [t for t in STATE.get("tables", []) if not page_filter or t["page"] in page_filter]
    if len(selected) < 2:
        return "", [], False, []

    q_tokens = {_STEMMER.stem(token) for token in re.findall(r"[a-z0-9]+", question.lower())}
    families = _ordered_relevant_families(selected, q_tokens)
    if len(families) < 2:
        return "", [], False, []

    relevant_signatures = {family["signature"] for family in families}
    occurrences = []
    for table in sorted(selected, key=lambda t: (t["page"], t["path"])):
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
    summaries = []

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


def _compose_join_analysis(question: str, page_filter: list[int] | None = None):
    base_analysis = _table_join_analysis(page_filter or None, question=question)
    inferred_ctx, inferred_pages, inferred_truncated, inferred_summaries = _build_inferred_join_context(
        question,
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
            f"({ _confidence_label(item['confidence']) } confidence, "
            f"row match {item['pair_count']} rows, ratio {item['row_ratio']:.2f})"
        )
    summary_lines.append(
        "These inferred joins are not exact-key joins. Use them only when clearly labeled as inferred."
    )
    return base_analysis + "\n" + "\n".join(summary_lines), inferred_ctx, inferred_pages, inferred_truncated


def _build_overview_context(page_filter: list[int] | None = None, char_limit: int = 18000):
    blocks = []
    total = 0
    used_pages: list[int] = []
    truncated = False
    for page_data in STATE.get("pages", []):
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


def _build_visual_context(page_filter: list[int] | None = None, char_limit: int = 12000):
    blocks = []
    total = 0
    used_pages: list[int] = []
    truncated = False
    for page_data in STATE.get("pages", []):
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


def _stream_direct_completion(prompt: str):
    response_text = ""
    for chunk in LLM.stream_complete(prompt):
        delta = getattr(chunk, "delta", "") or ""
        if not delta:
            text = getattr(chunk, "text", "") or ""
            delta = text[len(response_text):] if text.startswith(response_text) else text
        if not delta:
            continue
        response_text += delta
        yield response_text


# ── Chitchat guard ────────────────────────────────────────────────────────────
# Only for pure greetings — CondensePlusContextChatEngine handles everything
# else (including "what is this document?") via its retriever + system prompt.
_CHITCHAT_SET = {
    "hello", "hi", "hey", "howdy", "hiya", "sup", "wassup",
    "how are you", "how are u", "how r u",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thank u", "thx", "ty",
    "bye", "goodbye", "see you", "see ya",
    "who are you", "what are you", "are you an ai", "are you a bot",
    "ok", "okay", "cool", "great", "awesome", "yep", "nope", "yes", "no",
}
_DOC_WORDS = {
    "page", "document", "table", "name", "address", "animal", "phone",
    "show", "list", "what", "who", "tell", "find", "give", "how", "pdf",
    "image", "data", "register", "paper", "cv", "file", "uploaded",
}

def _is_chitchat(q: str) -> bool:
    ql = q.lower().strip().rstrip("?!.,").strip()
    if ql in _CHITCHAT_SET:
        return True
    # Short query with no document-intent words
    words = set(ql.split())
    if len(words) <= 3 and not (words & _DOC_WORDS):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def on_upload(file):
    """
    Generator: process one document file, yield status updates to the UI.

    Outputs (6 values):
      status_html, qa_col_visible, idle_html_visible, chatbot_reset,
      sources_reset, dl_btn_update
    """
    def emit(L, pct, stage, tone="run", show_qa=False, dl=gr.update(visible=False)):
        return (
            gr.update(value=render_log(L, pct, stage, tone), visible=True),
            gr.update(visible=show_qa),       # qa_col
            gr.update(visible=not show_qa),   # idle_html
            [],                               # reset chatbot
            "",                               # reset sources
            dl,                               # dl_btn
        )

    L = []
    if file is None:
        L.append(("dim", "Waiting for a file…"))
        yield emit(L, 0, "idle")
        return

    fname = os.path.basename(file.name if hasattr(file, "name") else str(file))
    fpath = file.name if hasattr(file, "name") else str(file)
    ext   = Path(fname).suffix.upper().lstrip(".")

    L.append(("inf", f'<span style="color:#6b7280">{_ts()}</span>  '
                     f'File: <b style="color:#e2e4eb">{_esc(fname)}</b>'))
    yield emit(L, 3, "decode")

    try:
        # ── 1. Route pages: digital parse first, OCR fallback ────────────────
        L.append(("run", f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'Routing {ext} content: native PDF parse first, OCR only where needed…'))
        yield emit(L, 6, "decode")
        all_pages = []
        all_tables = []
        routing = {"digital_pages": 0, "ocr_pages": 0, "native_table_pages": 0}
        n = 0

        for event in stream_document_extraction(fpath, fname):
            kind = event.get("event")
            if kind == "document_started":
                n = event.get("total_pages", 0)
                label = "page" if n == 1 else "pages"
                L.append(("dim", f'<span style="color:#6b7280">{_ts()}</span>  '
                                 f'<span style="color:#374151">{n} {label} detected</span>'))
                yield emit(L, 10, "decode")
                continue

            if kind == "page_started":
                page_num = event["page"]
                route = event.get("source", "unknown")
                route_label = "native text parser" if route == "digital-text" else "OCR fallback"
                L.append(("run", f'<span style="color:#6b7280">{_ts()}</span>  '
                                 f'Processing page <b style="color:#e2e4eb">{page_num}/{max(1, n)}</b> '
                                 f'via {route_label}…'))
                yield emit(L, 12 + int(((page_num - 1) / max(1, n)) * 34), f"read  {page_num}/{max(1, n)}")
                continue

            if kind == "page_complete":
                page_data = event["result"]
                all_pages.append(page_data)
                all_tables.extend(page_data.get("tables", []))
                tl = ""
                if page_data.get("tables"):
                    shapes = ", ".join(
                        f'{t["rows"]}r×{len(t["cols"])}c'
                        for t in page_data["tables"][:2]
                    )
                    extra = "" if len(page_data["tables"]) <= 2 else f' +{len(page_data["tables"]) - 2} more'
                    tl = f'  <span style="color:#f0a500">{len(page_data["tables"])} tbl</span> ({shapes}{extra})'
                src = page_data.get("source", "unknown")
                L.append(("ok", f'<span style="color:#6b7280">{_ts()}</span>  '
                                f'  ✓ Page {page_data["page"]} · <b style="color:#e2e4eb">'
                                f'{len(page_data.get("text", "")):,} chars</b> · {src}{tl}'))
                yield emit(
                    L,
                    14 + int((len(all_pages) / max(1, n)) * 38),
                    f"read  {len(all_pages)}/{max(1, n)}",
                )
                continue

            if kind == "complete":
                routing = event.get("routing", routing)
                n = event.get("total_pages", len(all_pages))

        L.append(("ok", f'<span style="color:#6b7280">{_ts()}</span>  ✓ {n} page(s) processed'))
        L.append((
            "dim",
            f'<span style="color:#6b7280">{_ts()}</span>  '
            f'<span style="color:#374151">Routing · native {routing.get("digital_pages", 0)} '
            f'· OCR {routing.get("ocr_pages", 0)} · native-table pages {routing.get("native_table_pages", 0)}</span>'
        ))
        yield emit(L, 54, "read")

        combined_text = "\n\n".join(
            _combined_page_text(page_data)
            for page_data in all_pages
            if _combined_page_text(page_data)
        )
        if not combined_text.strip():
            L.append(("err", f'<span style="color:#6b7280">{_ts()}</span>  '
                             "✗ No readable text or visual description extracted — check scan quality or try a higher-resolution scan"))
            yield emit(L, 0, "error", tone="err")
            return

        L.append(("dim", f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'<span style="color:#374151">'
                         f'Done · {len(combined_text):,} chars · {len(all_tables)} table(s)</span>'))

        # ── 3. Auto-summary ───────────────────────────────────────────────────
        L.append(("run", f'<span style="color:#6b7280">{_ts()}</span>  Generating document summary…'))
        yield emit(L, 66, "summarise")
        summary = auto_summarize(combined_text)
        L.append(("inf", f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'<span style="color:#60a5fa">Summary:</span> {_esc(summary[:90])}'))
        yield emit(L, 70, "summarise")

        # ── 4. Build LlamaIndex VectorStoreIndex ──────────────────────────────
        L.append(("run", f'<span style="color:#6b7280">{_ts()}</span>  Building index (BGE-small embeddings)…'))
        yield emit(L, 73, "embed")
        index = build_index(all_pages, all_tables, fname)
        L.append(("ok", f'<span style="color:#6b7280">{_ts()}</span>  '
                        f'✓ Index ready · {len(all_pages)} pages embedded'))
        yield emit(L, 88, "index")

        # ── 5. Build chat engine ──────────────────────────────────────────────
        L.append(("run", f'<span style="color:#6b7280">{_ts()}</span>  Building chat engine…'))
        yield emit(L, 90, "engine")
        engine = build_engine(index, summary, LLM)

        # ── 6. Zip tables for download ────────────────────────────────────────
        zip_path = make_zip(all_tables, fname)

        # ── Store in global state ─────────────────────────────────────────────
        STATE.update(
            engine=engine, tables=all_tables,
            name=fname, summary=summary, zip=zip_path,
            pages=all_pages, text=combined_text,
        )

        if all_tables:
            L.append(("ok", f'<span style="color:#6b7280">{_ts()}</span>  '
                            f'✓ {len(all_tables)} table(s) → CSV export ready'))
        L.append(("inf", f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'<span style="color:#22c55e">⬤ Ready</span>  — '
                         f'Chat unlocked for <b style="color:#e2e4eb">{_esc(fname)}</b>'))

        dl = gr.update(value=zip_path, visible=zip_path is not None)
        yield (
            gr.update(value=render_log(L, 100, "ready", "ok"), visible=True),
            gr.update(visible=True),    # qa_col
            gr.update(visible=False),   # idle_html
            [],
            "",
            dl,
        )

    except Exception as exc:
        traceback.print_exc()
        L.append(("err", f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'✗ {_esc(str(exc)[:120])}'))
        yield emit(L, 0, "error", tone="err")


# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════════

def on_transcribe(audio):
    if not audio:
        return gr.update()
    text = transcribe_audio(audio)
    return gr.update(value=text) if text else gr.update()


# ══════════════════════════════════════════════════════════════════════════════
# Q&A HANDLER
# ══════════════════════════════════════════════════════════════════════════════

THINKING_MSGS = [
    "🤔 Condensing question…",
    "🔍 Retrieving relevant passages…",
    "📊 Checking extracted tables…",
    "✍️ Composing answer…",
]


def on_ask(question: str, history: list, msg_count: int):
    """
    Main Q&A generator.

    Uses LlamaIndex's CondensePlusContextChatEngine exclusively.

    The engine internally:
      1. Condenses conversation + question → standalone query (handles pronouns,
         "the second one", "what about her address?" etc.)
      2. Retrieves TOP_K nodes via VectorStoreIndex
      3. Builds response using context + full chat history (ChatSummaryMemoryBuffer)
      4. Automatically stores the turn in memory

    We only need to: call stream_chat → stream tokens → display source_nodes.

    Outputs: chatbot, sources_html, question_in, history_panel, msg_count_state
    """
    if not question or not question.strip():
        yield history, "", gr.update(), gr.update(), msg_count
        return

    history    = list(history or []) + [[question, None]]
    msg_count  = (msg_count or 0) + 1

    # ── Message limit: hard reset at 20 ──────────────────────────────────────
    if msg_count >= MSG_LIMIT:
        engine = STATE.get("engine")
        if engine:
            try:
                engine.reset()
            except Exception:
                pass
        history = [[question,
                    "⚠️ Context reset after 20 messages. Conversation history cleared. "
                    "Ask your next question as if starting fresh."]]
        src = render_context_cleared_badge()
        yield history, src, gr.update(value=""), render_history_html(), 0
        return

    # ── Soft warning at MSG_WARN_AT ───────────────────────────────────────────
    show_warning = (msg_count == MSG_WARN_AT)
    remaining    = MSG_LIMIT - msg_count

    # ── Chitchat: skip retrieval entirely ─────────────────────────────────────
    if _is_chitchat(question):
        history[-1][1] = "💬 Chatting…"
        yield history, "", gr.update(value=""), gr.update(), msg_count

        # Use a SimpleChatEngine for greetings to avoid unnecessary retrieval latency
        from llama_index.core.chat_engine import SimpleChatEngine
        simple = SimpleChatEngine.from_defaults(
            llm=LLM,
            system_prompt=(
                "You are a friendly document assistant. Respond warmly in 1-2 sentences."
                + (f" A document is loaded: {STATE['summary']}." if STATE.get("summary") else "")
            ),
        )
        response_text = ""
        try:
            sr = simple.stream_chat(question)
            for token in sr.response_gen:
                response_text += token
                history[-1][1] = response_text
                yield history, "", gr.update(), gr.update(), msg_count
        except Exception:
            history[-1][1] = "Hello! How can I help you with the document?"
            yield history, "", gr.update(), gr.update(), msg_count

        src = render_chitchat_badge()
        if show_warning:
            src = render_warn_badge(remaining) + src
        save_turn(STATE.get("name", ""), question, response_text, "chitchat")
        yield history, src, gr.update(), render_history_html(), msg_count
        return

    # ── No document loaded → general knowledge ─────────────────────────────
    if STATE.get("engine") is None:
        history[-1][1] = "🧠 No document loaded — using general knowledge…"
        yield history, "", gr.update(value=""), gr.update(), msg_count

        from llama_index.core.chat_engine import SimpleChatEngine
        simple = SimpleChatEngine.from_defaults(llm=LLM)
        response_text = ""
        try:
            sr = simple.stream_chat(question)
            for token in sr.response_gen:
                response_text += token
                history[-1][1] = response_text
                yield history, "", gr.update(), gr.update(), msg_count
        except Exception as e:
            history[-1][1] = f"⚠️ Error: {e}"
            yield history, "", gr.update(), gr.update(), msg_count
            return

        src = render_general_badge()
        if show_warning:
            src = render_warn_badge(remaining) + src
        save_turn(STATE.get("name", "(no doc)"), question, response_text, "general")
        yield history, src, gr.update(), render_history_html(), msg_count
        return

    # ── Document loaded → planned route ───────────────────────────────────────
    engine = STATE["engine"]
    page_filter = _extract_page_numbers(question)

    if _is_visual_query(question):
        history[-1][1] = "👁 Reading the visual content…"
        yield history, "", gr.update(value=""), gr.update(), msg_count

        context, used_pages, truncated = _build_visual_context(page_filter or None)
        if context:
            prompt = VISUAL_QA_PROMPT_TEMPLATE.format(
                doc_summary=STATE.get("summary", "Uploaded document"),
                history=_recent_history_for_prompt(engine),
                context=context,
                question=question,
                format_guidance=_format_guidance(question),
            )

            response_text = ""
            try:
                for response_text in _stream_direct_completion(prompt):
                    history[-1][1] = response_text
                    yield history, "", gr.update(), gr.update(), msg_count
            except Exception as e:
                history[-1][1] = f"⚠️ Error: {e}"
                traceback.print_exc()
                yield history, "", gr.update(), gr.update(), msg_count
                return

            _remember_direct_turn(engine, question, response_text)
            src = render_visual_sources(used_pages, truncated)
            if show_warning:
                src = render_warn_badge(remaining) + src
            save_turn(STATE.get("name", ""), question, response_text, "document-visual")
            yield history, src, gr.update(), render_history_html(), msg_count
            return

    if _is_overview_query(question):
        history[-1][1] = "🧭 Reading the document overview…"
        yield history, "", gr.update(value=""), gr.update(), msg_count

        context, used_pages, truncated = _build_overview_context(page_filter or None)
        if not context:
            context = (STATE.get("text") or "")[:6000]
            used_pages = page_filter or [p["page"] for p in STATE.get("pages", [])[:3]]
            truncated = len(STATE.get("text") or "") > len(context)

        prompt = OVERVIEW_PROMPT_TEMPLATE.format(
            doc_summary=STATE.get("summary", "Uploaded document"),
            history=_recent_history_for_prompt(engine),
            context=context or "(no document text available)",
            question=question,
            format_guidance=_format_guidance(question),
        )

        response_text = ""
        try:
            for response_text in _stream_direct_completion(prompt):
                history[-1][1] = response_text
                yield history, "", gr.update(), gr.update(), msg_count
        except Exception as e:
            history[-1][1] = f"⚠️ Error: {e}"
            traceback.print_exc()
            yield history, "", gr.update(), gr.update(), msg_count
            return

        _remember_direct_turn(engine, question, response_text)
        src = render_overview_sources(used_pages, truncated)
        if show_warning:
            src = render_warn_badge(remaining) + src
        save_turn(STATE.get("name", ""), question, response_text, "document-overview")
        yield history, src, gr.update(), render_history_html(), msg_count
        return

    if _is_relationship_query(question):
        history[-1][1] = "🔗 Checking whether values from different table groups can be safely linked…"
        yield history, "", gr.update(value=""), gr.update(), msg_count

        full_ctx, used_pages, truncated = _build_table_context(page_filter or None)
        if full_ctx:
            join_analysis, inferred_ctx, inferred_pages, inferred_truncated = _compose_join_analysis(
                question,
                page_filter or None,
            )
            context = full_ctx
            if inferred_ctx:
                context = inferred_ctx + "\n\n[RAW TABLES]\n" + full_ctx
                for page_num in inferred_pages:
                    if page_num not in used_pages:
                        used_pages.append(page_num)
                truncated = truncated or inferred_truncated
            prompt = RELATIONSHIP_PROMPT_TEMPLATE.format(
                history=_recent_history_for_prompt(engine),
                question=question,
                format_guidance=_format_guidance(question),
                join_analysis=join_analysis,
                context=context,
            )

            response_text = ""
            try:
                for response_text in _stream_direct_completion(prompt):
                    history[-1][1] = response_text
                    yield history, "", gr.update(), gr.update(), msg_count
            except Exception as e:
                history[-1][1] = f"⚠️ Error: {e}"
                traceback.print_exc()
                yield history, "", gr.update(), gr.update(), msg_count
                return

            _remember_direct_turn(engine, question, response_text)
            src = render_full_table_sources(STATE["tables"], used_pages, truncated)
            if show_warning:
                src = render_warn_badge(remaining) + src
            save_turn(STATE.get("name", ""), question, response_text, "document-relationship")
            yield history, src, gr.update(), render_history_html(), msg_count
            return

    if _is_synthesis_query(question):
        history[-1][1] = "📊 Combining the extracted tables…"
        yield history, "", gr.update(value=""), gr.update(), msg_count

        full_ctx, used_pages, truncated = _build_table_context(page_filter or None)
        if full_ctx:
            join_analysis, inferred_ctx, inferred_pages, inferred_truncated = _compose_join_analysis(
                question,
                page_filter or None,
            )
            context = full_ctx
            if inferred_ctx:
                context = inferred_ctx + "\n\n[RAW TABLES]\n" + full_ctx
                for page_num in inferred_pages:
                    if page_num not in used_pages:
                        used_pages.append(page_num)
                truncated = truncated or inferred_truncated
            prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                history=_recent_history_for_prompt(engine),
                question=question,
                format_guidance=_format_guidance(question),
                join_analysis=join_analysis,
                context=context,
            )

            response_text = ""
            try:
                for response_text in _stream_direct_completion(prompt):
                    history[-1][1] = response_text
                    yield history, "", gr.update(), gr.update(), msg_count
            except Exception as e:
                history[-1][1] = f"⚠️ Error: {e}"
                traceback.print_exc()
                yield history, "", gr.update(), gr.update(), msg_count
                return

            _remember_direct_turn(engine, question, response_text)
            src = render_full_table_sources(STATE["tables"], used_pages, truncated)
            if show_warning:
                src = render_warn_badge(remaining) + src
            save_turn(STATE.get("name", ""), question, response_text, "document-synthesis")
            yield history, src, gr.update(), render_history_html(), msg_count
            return

    # ── Standard document QA → CondensePlusContextChatEngine ─────────────────
    history[-1][1] = THINKING_MSGS[0]
    yield history, "", gr.update(value=""), gr.update(), msg_count
    time.sleep(0.12)
    history[-1][1] = THINKING_MSGS[1]
    yield history, "", gr.update(), gr.update(), msg_count

    response_text  = ""
    streaming_resp = None

    try:
        streaming_resp = engine.stream_chat(_apply_format_guidance(question))
        for token in streaming_resp.response_gen:
            response_text += token
            history[-1][1] = response_text
            yield history, "", gr.update(), gr.update(), msg_count
    except Exception as e:
        history[-1][1] = f"⚠️ Error: {e}"
        traceback.print_exc()
        yield history, "", gr.update(), gr.update(), msg_count
        return

    # ── Source panel ──────────────────────────────────────────────────────────
    source_nodes = getattr(streaming_resp, "source_nodes", []) or []
    src = render_sources(source_nodes, STATE["tables"])
    if show_warning:
        src = render_warn_badge(remaining) + src

    save_turn(STATE.get("name", ""), question, response_text, "document")
    yield history, src, gr.update(), render_history_html(), msg_count


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,700;0,9..144,900;1,9..144,300&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

.progress-bar-wrap,.progress-level,.progress-level-inner,
.meta-text,.meta-text-center,.eta-bar,.loader,
div.progress-bar,.pending-bar {{ display:none !important; }}
.generating {{ border-color:transparent !important; animation:none !important; }}

:root {{
    --bg:#0d0f14; --sur:#13151c; --sur2:#191c25; --bdr:#23273a;
    --text:#dde1ec; --muted:#5a6278; --amber:#f0a500; --green:#22c55e; --r:11px;
}}
*,*::before,*::after {{ box-sizing:border-box; }}

.gradio-container {{
    font-family:'Inter',system-ui,sans-serif !important;
    background:var(--bg) !important; color:var(--text) !important;
    max-width:100% !important; padding:0 !important;
}}
footer {{ display:none !important; }}
.gradio-container label, .gradio-container .label-wrap > span,
.gradio-container p {{ color:var(--text) !important; }}

.gradio-container textarea, .gradio-container input[type=text] {{
    background:var(--sur2) !important; border:1px solid var(--bdr) !important;
    color:var(--text) !important; border-radius:8px !important; font-size:14px !important;
}}
.gradio-container textarea:focus, .gradio-container input:focus {{
    border-color:var(--amber) !important;
    box-shadow:0 0 0 3px rgba(240,165,0,.1) !important; outline:none !important;
}}
.gradio-container button.primary {{
    background:var(--amber) !important; color:#0d0f14 !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important;
    font-size:14px !important; transition:opacity .15s,transform .1s !important;
}}
.gradio-container button.primary:hover  {{ opacity:.88 !important; transform:translateY(-1px) !important; }}
.gradio-container button:not(.primary) {{
    background:var(--sur2) !important; border:1px solid var(--bdr) !important;
    color:var(--text) !important; border-radius:8px !important;
}}
.gradio-container .file-preview,
.gradio-container [data-testid="file"] {{
    background:var(--sur2) !important; border:1.5px dashed var(--bdr) !important;
    border-radius:var(--r) !important; color:var(--muted) !important;
}}
.gradio-container [data-testid="file"]:hover {{ border-color:var(--amber) !important; }}
.gradio-container [data-testid="audio"] {{
    background:var(--sur2) !important; border:1px solid var(--bdr) !important; border-radius:var(--r) !important;
}}

/* Chat bubbles */
.gradio-container .chatbot {{ background:transparent !important; border:none !important; }}
.gradio-container .message-bubble-border {{ box-shadow:none !important; }}
.gradio-container .message.user {{
    justify-content:flex-end !important; margin-bottom:18px !important;
}}
.gradio-container .message.user .message-bubble-border {{
    background:#15171f !important; border:none !important;
    border-right:3px solid var(--amber) !important;
    border-radius:10px 2px 2px 10px !important;
    padding:9px 14px 9px 12px !important; max-width:68% !important;
    color:#c8cde0 !important; font-size:13.5px !important; line-height:1.65 !important;
}}
.gradio-container .message.bot {{
    justify-content:flex-start !important; margin-bottom:18px !important;
}}
.gradio-container .message.bot .message-bubble-border {{
    background:transparent !important; border:none !important;
    border-left:2px solid #2a2e42 !important; border-radius:0 !important;
    padding:6px 14px !important; max-width:92% !important;
    color:#c8cde0 !important; font-size:13.5px !important; line-height:1.8 !important;
    transition:border-left-color .2s ease !important; overflow-x:auto !important;
}}
.gradio-container .message.bot:hover .message-bubble-border {{
    border-left-color:var(--amber) !important;
}}
/* Markdown tables inside chat */
.gradio-container .message.bot .message-bubble-border table {{
    border-collapse:collapse !important; width:100% !important;
    margin:10px 0 !important; font-size:12.5px !important;
    font-family:'JetBrains Mono',monospace !important;
}}
.gradio-container .message.bot .message-bubble-border th {{
    background:#1e2130 !important; color:#f0a500 !important;
    padding:6px 12px !important; border:1px solid #2a2e42 !important;
    text-align:left !important; font-weight:600 !important;
}}
.gradio-container .message.bot .message-bubble-border td {{
    background:#13151c !important; color:#c8cde0 !important;
    padding:5px 12px !important; border:1px solid #1e2130 !important;
}}
.gradio-container .message.bot .message-bubble-border tr:hover td {{
    background:#191c25 !important;
}}
.gradio-container .message .avatar-container,
.gradio-container .avatar-container,
.gradio-container .icon-wrap {{ display:none !important; }}

/* Layout */
#root {{ max-width:1360px; margin:0 auto; padding:28px 22px 60px; }}
.hero {{
    display:flex; align-items:flex-end; justify-content:space-between;
    gap:20px; flex-wrap:wrap; padding-bottom:24px; margin-bottom:26px;
    border-bottom:1px solid var(--bdr);
}}
.hero-eyebrow {{
    font-family:'JetBrains Mono',monospace; font-size:10px;
    font-weight:500; letter-spacing:.16em; text-transform:uppercase;
    color:var(--amber); margin-bottom:8px;
}}
.hero-title {{
    font-family:'Fraunces',serif; font-size:36px; font-weight:900;
    letter-spacing:-.025em; line-height:1; color:var(--text); margin:0;
}}
.hero-title em {{ font-style:italic; color:var(--amber); }}
.hero-sub {{ margin:8px 0 0; font-size:13.5px; color:var(--muted); line-height:1.55; }}
.hero-right {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
.pill {{
    display:inline-flex; align-items:center; gap:7px;
    background:var(--sur); border:1px solid var(--bdr); border-radius:999px;
    padding:5px 12px; font-family:'JetBrains Mono',monospace; font-size:11px;
    color:var(--muted); white-space:nowrap;
}}
.dot-g {{ width:7px;height:7px;border-radius:50%;background:var(--green);
          box-shadow:0 0 6px var(--green);animation:pulse 2.3s ease-in-out infinite; }}
.dot-b {{ width:7px;height:7px;border-radius:50%;background:{BADGE_COLOR};
          box-shadow:0 0 6px {BADGE_COLOR}; }}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.panel {{ background:var(--sur);border:1px solid var(--bdr);border-radius:12px;
          padding:18px;overflow:hidden; }}
.plabel {{
    display:block;font-size:10px;font-weight:600;letter-spacing:.14em;
    text-transform:uppercase;font-family:'JetBrains Mono',monospace;
    color:var(--muted);padding-bottom:12px;margin-bottom:16px;
    border-bottom:1px solid var(--bdr);
}}
.stages {{ display:flex; flex-direction:column; }}
.stage {{ display:flex;align-items:center;gap:11px;padding:7px 0;
          border-bottom:1px solid var(--sur2); }}
.stage:last-child {{ border-bottom:none; }}
.snum {{ width:20px;height:20px;border-radius:50%;background:var(--sur2);
         border:1px solid var(--bdr);display:flex;align-items:center;
         justify-content:center;font-size:9px;font-family:'JetBrains Mono',monospace;
         color:var(--muted);flex-shrink:0; }}
.sname {{ font-size:13px;font-weight:600;color:var(--text); }}
.sdesc {{ font-size:11.5px;color:var(--muted);margin-top:1px; }}
.idle {{ display:flex;flex-direction:column;align-items:center;justify-content:center;
         min-height:340px;gap:12px;text-align:center;padding:40px 24px; }}
.idle-txt {{ font-size:13.5px;color:var(--muted);max-width:220px;line-height:1.65; }}
.src-panel {{ max-height:420px;overflow-y:auto;padding:4px 0;
              scrollbar-width:thin;scrollbar-color:var(--bdr) transparent; }}
.src-panel::-webkit-scrollbar {{ width:5px; }}
.src-panel::-webkit-scrollbar-thumb {{ background:var(--bdr);border-radius:3px; }}
.hist-panel {{ max-height:200px;overflow-y:auto;padding:2px 0;
               scrollbar-width:thin;scrollbar-color:var(--bdr) transparent; }}
.hist-panel::-webkit-scrollbar {{ width:4px; }}
.hist-panel::-webkit-scrollbar-thumb {{ background:var(--bdr);border-radius:3px; }}
.hist-panel details summary::-webkit-details-marker {{ display:none; }}
.input-row {{ display:flex;gap:8px;align-items:flex-end; }}
.divider {{ height:1px;background:var(--bdr);margin:14px 0; }}
@media(max-width:700px){{
    .hero{{flex-direction:column;align-items:flex-start;}}
    .hero-title{{font-size:26px;}}
    #root{{padding:16px 12px 40px;}}
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
    ),
    css=CSS, title="Document Q&A",
) as app:

    msg_count_state = gr.State(0)

    gr.HTML(f"""
    <div id="root">
      <div class="hero">
        <div>
          <div class="hero-eyebrow">Document Intelligence · LlamaIndex</div>
          <h1 class="hero-title">Read. Search. <em>Answer.</em></h1>
          <p class="hero-sub">
            Upload any document — scanned, handwritten, printed, or a research paper.<br>
            Powered by LlamaIndex · CondensePlusContext · ChatSummaryMemory.
          </p>
        </div>
        <div class="hero-right">
          <span class="pill"><span class="dot-g"></span>System ready</span>
          <span class="pill"><span class="dot-b"></span>{BACKEND_LABEL}</span>
          <span class="pill">LlamaIndex · BGE-small</span>
          <span class="pill">20-turn memory</span>
        </div>
      </div>
    </div>""")

    with gr.Row(elem_id="root", equal_height=False):

        # ── LEFT: Upload + Pipeline + History ─────────────────────────────────
        with gr.Column(scale=4, min_width=280, elem_classes="panel"):
            gr.HTML('<span class="plabel">01 · Upload Document</span>')

            file_in = gr.File(
                label="Drop a PDF, PNG, JPG, or TIFF — any document type",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff"],
                height=120,
            )
            process_btn = gr.Button("Analyse Document", variant="primary", size="lg")
            status_box  = gr.HTML(visible=False)
            dl_btn      = gr.File(
                label="Download extracted tables (CSV zip)",
                visible=False, interactive=False,
            )

            gr.HTML("""
            <div class="divider"></div>
            <div class="stages">
              <div class="stage">
                <div class="snum">1</div>
                <div><div class="sname">Decode</div>
                     <div class="sdesc">Rasterise at 200 DPI</div></div>
              </div>
              <div class="stage">
                <div class="snum">2</div>
                <div><div class="sname">Read</div>
                     <div class="sdesc">Universal OCR · tables → CSV</div></div>
              </div>
              <div class="stage">
                <div class="snum">3</div>
                <div><div class="sname">Summarise</div>
                     <div class="sdesc">Auto one-sentence description</div></div>
              </div>
              <div class="stage">
                <div class="snum">4</div>
                <div><div class="sname">Index</div>
                     <div class="sdesc">LlamaIndex · SentenceSplitter · BGE</div></div>
              </div>
              <div class="stage">
                <div class="snum">5</div>
                <div><div class="sname">Ready</div>
                     <div class="sdesc">CondensePlusContext chat unlocked</div></div>
              </div>
            </div>
            <div class="divider"></div>""")

            gr.HTML(
                '<span style="display:block;font-size:10px;font-weight:600;'
                'letter-spacing:.14em;text-transform:uppercase;'
                'font-family:\'JetBrains Mono\',monospace;color:#374151;'
                'margin-bottom:10px">Past Sessions</span>'
            )
            history_panel = gr.HTML(
                value=render_history_html(), elem_classes="hist-panel"
            )

        # ── RIGHT: Chat ────────────────────────────────────────────────────────
        with gr.Column(scale=7, elem_classes="panel"):
            gr.HTML('<span class="plabel">02 · Chat</span>')

            idle_html = gr.HTML("""
            <div class="idle">
              <div style="opacity:.12">
                <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                  <rect x="1.5" y="1.5" width="49" height="49" rx="10"
                        stroke="#dde1ec" stroke-width="1.2"/>
                  <path d="M15 19h22M15 26h15M15 33h10" stroke="#dde1ec"
                        stroke-width="1.2" stroke-linecap="round"/>
                </svg>
              </div>
              <div class="idle-txt">
                Upload a document to unlock chat.<br><br>
                Supports handwritten registers, research papers,<br>
                CVs, invoices, scanned forms, and images.
              </div>
            </div>""", visible=True)

            qa_col = gr.Column(visible=False)
            with qa_col:
                chatbot = gr.Chatbot(
                    label="", height=460,
                    show_copy_button=True,
                    show_label=False,
                    render_markdown=True,
                    sanitize_html=False,
                    type="tuples",
                )

                with gr.Row(elem_classes="input-row"):
                    question_in = gr.Textbox(
                        label="",
                        placeholder="Ask anything · follow-up questions work naturally · Enter to send",
                        lines=1, scale=8, container=False,
                    )
                    ask_btn = gr.Button("↑", variant="primary", scale=1, min_width=48)

                audio_in = gr.Audio(
                    label="🎤  Voice — transcription appears in the box above",
                    type="filepath", sources=["microphone"], max_length=30,
                )

                gr.HTML('<div class="divider"></div>')
                gr.HTML(
                    '<span style="font-size:10px;font-family:monospace;color:#374151;'
                    'letter-spacing:.1em;text-transform:uppercase">'
                    'Sources &amp; retrieved tables</span>'
                )
                sources_out = gr.HTML("", elem_classes="src-panel")

    # ── Event wiring ────────────────────────────────────────────────────────────

    process_btn.click(
        fn=on_upload,
        inputs=[file_in],
        outputs=[status_box, qa_col, idle_html, chatbot, sources_out, dl_btn],
        queue=True, show_progress="hidden",
    )

    audio_in.stop_recording(
        fn=on_transcribe,
        inputs=[audio_in],
        outputs=[question_in],
        queue=True,
    )

    _ask_cfg = dict(
        fn=on_ask,
        inputs=[question_in, chatbot, msg_count_state],
        outputs=[chatbot, sources_out, question_in, history_panel, msg_count_state],
        queue=True,
        show_progress="hidden",
    )
    ask_btn.click(**_ask_cfg)
    question_in.submit(**_ask_cfg)


if __name__ == "__main__":
    print("Launching docqa v3 (LlamaIndex)…")
    app.queue(default_concurrency_limit=2)
    app.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        share=GRADIO_SHARE,
        inbrowser=OPEN_BROWSER,
        allowed_paths=[WORK_DIR],
    )
