"""
ocr.py — Document loading, Vision OCR, and table extraction.

This is the custom layer that feeds LlamaIndex.
Everything here stays hand-rolled because no framework handles:
  - pdf2image rasterisation
  - Vision-model OCR of handwritten/scanned pages
  - Pipe-table parsing → CSV files
  - Table row → natural-language sentence conversion

Public API:
  file_to_images(path)  -> list[PIL.Image]
  ocr_page(image, page_num, doc_name) -> dict(text, tables, page)
  stream_document_extraction(path, doc_name) -> iterator[dict]
  extract_document(path, doc_name) -> dict(pages, tables, routing)
  tables_to_row_chunks(tables, doc_name) -> list[str]
  make_zip(tables) -> str | None
  auto_summarize(text) -> str
"""

import os, io, re, base64, time, zipfile, mimetypes
from pathlib import Path
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

from config import WORK_DIR, DPI, MAX_PAGES, BACKEND, OCR_TOKENS, QA_TOKENS
from prompts import OCR_PROMPT, SUMMARIZE_PROMPT, VISUAL_SUMMARY_PROMPT


# ══════════════════════════════════════════════════════════════════════════════
# PAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def pdf_to_images(path: str) -> list:
    from pdf2image import convert_from_path
    pages = convert_from_path(
        path, dpi=DPI, first_page=1, last_page=MAX_PAGES,
        fmt="JPEG", thread_count=2,
    )
    print(f"PDF: {len(pages)} page(s) at {DPI} DPI")
    return pages


def load_image_file(path: str) -> list:
    img = _normalize_image(Image.open(path))
    print(f"Image: {Path(path).name}")
    return [img]


def file_to_images(path: str) -> list:
    """Dispatch PDF or image file to the right loader."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return pdf_to_images(path)
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"):
        return load_image_file(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _img_to_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def _enhance_for_retry(image: Image.Image) -> Image.Image:
    img = _normalize_image(image)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return img


def _text_signal_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]", text or ""))


def _is_image_upload(doc_name: str) -> bool:
    return Path(doc_name).suffix.lower() in {
        ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"
    }


def _should_retry_sparse_ocr(text: str, tables: list) -> bool:
    if tables:
        return False
    return _text_signal_count(text) < 80


def _should_capture_visual_summary(doc_name: str, text: str, tables: list) -> bool:
    if _is_image_upload(doc_name):
        return True
    if tables:
        return False
    return _text_signal_count(text) < 180


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _digital_text_confidence(text: str) -> float:
    text = _normalize_text(text)
    if not text:
        return 0.0
    words = text.split()
    if len(words) < 20:
        return 0.2
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, len(text))
    space_ratio = text.count(" ") / max(1, len(text))
    weird_ratio = sum(ch in {"�", "\u25a1"} for ch in text) / max(1, len(text))
    score = 0.45
    score += min(len(words) / 350, 0.25)
    score += min(alpha_ratio, 0.2)
    score += 0.08 if 0.08 <= space_ratio <= 0.28 else 0.0
    score -= weird_ratio * 2
    return max(0.0, min(score, 1.0))


def _choose_native_text(layout_text: str, plain_text: str) -> str:
    layout_score = _text_signal_count(layout_text)
    plain_score = _text_signal_count(plain_text)
    if layout_score >= plain_score:
        return _normalize_text(layout_text)
    return _normalize_text(plain_text)


def _render_pdf_page(path: str, page_num: int) -> Image.Image:
    from pdf2image import convert_from_path

    pages = convert_from_path(
        path,
        dpi=DPI,
        first_page=page_num,
        last_page=page_num,
        fmt="PNG",
        thread_count=1,
    )
    return _normalize_image(pages[0])


def _looks_like_table_row(parts: list[str]) -> bool:
    if len(parts) < 3:
        return False
    dense_long_parts = sum(len(part) > 90 for part in parts)
    if dense_long_parts >= 2:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# OCR BACKENDS  (dual: OpenAI vision / Gemini vision)
# ══════════════════════════════════════════════════════════════════════════════

if BACKEND == "openai":
    from openai import OpenAI as _OA
    from config import OPENAI_KEY, OCR_MODEL
    _oa_client = _OA(api_key=OPENAI_KEY)

    def _raw_ocr(image: Image.Image, page_num: int) -> str:
        b64 = _img_to_b64(image)
        for attempt in range(3):
            try:
                r = _oa_client.chat.completions.create(
                    model=OCR_MODEL, max_tokens=OCR_TOKENS, temperature=0.1,
                    messages=[{"role": "user", "content": [
                        {"type": "text",      "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    ]}])
                return r.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                w = 10 * (attempt + 1)
                if "429" in err or "rate" in err.lower():
                    print(f"  Rate limit p{page_num}, wait {w}s"); time.sleep(w)
                else:
                    print(f"  OCR err p{page_num} attempt {attempt+1}: {err[:80]}"); time.sleep(3)
        return f"[OCR failed page {page_num}]"

    def _visual_call(image: Image.Image, page_num: int) -> str:
        b64 = _img_to_b64(image)
        for attempt in range(2):
            try:
                r = _oa_client.chat.completions.create(
                    model=OCR_MODEL, max_tokens=256, temperature=0.1,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": VISUAL_SUMMARY_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    ]}])
                return r.choices[0].message.content.strip()
            except Exception as e:
                print(f"  Visual summary err p{page_num} attempt {attempt+1}: {str(e)[:80]}")
                time.sleep(2)
        return ""

    def _quick_call(prompt: str, max_tokens: int = 80) -> str:
        """Cheap LLM call for summarisation."""
        r = _oa_client.chat.completions.create(
            model=OCR_MODEL, max_tokens=max_tokens, temperature=0.1,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()

    def transcribe_audio(audio_path: str) -> str:
        if not audio_path or not Path(audio_path).exists(): return ""
        try:
            with open(audio_path, "rb") as f:
                r = _oa_client.audio.transcriptions.create(
                    model="whisper-1", file=f, response_format="text")
            return (r if isinstance(r, str) else r.text).strip()
        except Exception as e:
            print(f"  Transcribe err: {e}"); return ""

else:
    import google.generativeai as genai
    from config import GEMINI_KEY, GEMINI_MODEL
    genai.configure(api_key=GEMINI_KEY)
    _gem_model = genai.GenerativeModel(GEMINI_MODEL)

    def _raw_ocr(image: Image.Image, page_num: int) -> str:
        for attempt in range(3):
            try:
                r = _gem_model.generate_content(
                    [OCR_PROMPT, image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1, max_output_tokens=OCR_TOKENS))
                return r.text.strip()
            except Exception as e:
                err = str(e); w = 15 * (attempt + 1)
                if "429" in err or "quota" in err.lower():
                    print(f"  Rate limit p{page_num}, wait {w}s"); time.sleep(w)
                else:
                    print(f"  OCR err p{page_num} attempt {attempt+1}: {err[:80]}"); time.sleep(3)
        return f"[OCR failed page {page_num}]"

    def _visual_call(image: Image.Image, page_num: int) -> str:
        for attempt in range(2):
            try:
                r = _gem_model.generate_content(
                    [VISUAL_SUMMARY_PROMPT, image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1, max_output_tokens=256))
                return r.text.strip()
            except Exception as e:
                print(f"  Visual summary err p{page_num} attempt {attempt+1}: {str(e)[:80]}")
                time.sleep(2)
        return ""

    def _quick_call(prompt: str, max_tokens: int = 80) -> str:
        r = _gem_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, max_output_tokens=max_tokens))
        return r.text.strip()

    def transcribe_audio(audio_path: str) -> str:
        if not audio_path or not Path(audio_path).exists(): return ""
        try:
            ab   = Path(audio_path).read_bytes()
            if len(ab) > 20 * 1024 * 1024: return ""
            mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
            b64  = base64.b64encode(ab).decode()
            r    = _gem_model.generate_content([
                "Transcribe the speech in this audio. Output only the transcribed text.",
                {"inline_data": {"mime_type": mime, "data": b64}},
            ])
            return r.text.strip()
        except Exception as e:
            print(f"  Transcribe err: {e}"); return ""


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PARSING  (pipe-separated OCR output → CSV files)
# ══════════════════════════════════════════════════════════════════════════════

def _flush_table(rows: list, page_num: int, doc_name: str, tables: list):
    """Convert a buffer of pipe-split rows into a clean CSV + metadata entry."""
    headers, data = rows[0], rows[1:]
    padded = []
    for r in data:
        if len(r) < len(headers):   r = r + [""] * (len(headers) - len(r))
        elif len(r) > len(headers): r = r[:len(headers)]
        padded.append(r)
    if not padded:
        return
    df = pd.DataFrame(padded, columns=headers)
    # Drop columns that are entirely empty
    df = df.replace("", pd.NA).dropna(axis=1, how="all").fillna("")
    safe_name = re.sub(r"[^\w]", "_", doc_name)[:30]
    path = os.path.join(WORK_DIR, f"{safe_name}_p{page_num}_{len(tables)}.csv")
    df.to_csv(path, index=False)
    tables.append({
        "page": page_num, "path": path,
        "rows": len(df), "cols": list(df.columns),
    })
    print(f"    Table: {len(df)}r × {len(df.columns)}c → {os.path.basename(path)}")


def parse_pipe_tables(text: str, page_num: int, doc_name: str) -> list:
    """
    Scan OCR text for pipe-separated table blocks, save each as a CSV.

    Threshold ≥ 2 pipe-parts (not 3) so rightmost columns like
    'Kind of Animal' or 'Signature' survive even if OCR drops the last pipe.
    """
    tables, buf = [], []
    for line in text.split("\n"):
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2:
            buf.append(parts)
        else:
            if len(buf) >= 2:
                _flush_table(buf, page_num, doc_name, tables)
            buf = []
    if len(buf) >= 2:
        _flush_table(buf, page_num, doc_name, tables)
    return tables


def parse_layout_tables(text: str, page_num: int, doc_name: str) -> list:
    """
    Parse native PDF layout text into CSVs by treating 2+ spaces as column gaps.

    This is a lightweight native-table path for born-digital PDFs, so we do not
    OCR clean pages just to recover obvious tables.
    """
    tables, buf = [], []
    expected_cols = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if len(buf) >= 2:
                _flush_table(buf, page_num, doc_name, tables)
            buf = []
            expected_cols = None
            continue

        parts = [p.strip() for p in re.split(r"\s{2,}", stripped) if p.strip()]
        if not _looks_like_table_row(parts):
            if len(buf) >= 2:
                _flush_table(buf, page_num, doc_name, tables)
            buf = []
            expected_cols = None
            continue

        if expected_cols is None:
            expected_cols = len(parts)
            buf.append(parts)
            continue

        if len(parts) >= max(3, expected_cols - 1):
            buf.append(parts)
            expected_cols = max(expected_cols, len(parts))
        else:
            if len(buf) >= 2:
                _flush_table(buf, page_num, doc_name, tables)
            buf = [parts]
            expected_cols = len(parts)

    if len(buf) >= 2:
        _flush_table(buf, page_num, doc_name, tables)
    return tables


def tables_to_row_chunks(tables: list, doc_name: str) -> list:
    """
    Convert every saved CSV row into a self-contained natural-language sentence.

    Example:
      "[Doc: register.pdf][Page 1][TABLE ROW] Owner: M. Abid | Address: 288/HR
       | Mobile: 0333-3796280 | Kind of Animal: Calf"

    These sentences land in the LlamaIndex VectorStoreIndex alongside regular
    text chunks. A query like "what animals are on page 1?" finds them directly
    because the page tag, column name, and value all appear together.
    """
    chunks = []
    for t in tables:
        try:
            df = pd.read_csv(t["path"])
            for _, row in df.iterrows():
                parts = " | ".join(
                    f"{col}: {val}"
                    for col, val in zip(df.columns, row)
                    if str(val).strip() and str(val).strip().lower() not in ("nan", "")
                )
                if parts:
                    chunks.append(
                        f"[Doc: {doc_name}][Page {t['page']}][TABLE ROW] {parts}"
                    )
        except Exception:
            pass
    return chunks


def make_zip(tables: list, doc_name: str) -> str | None:
    """Bundle all CSVs for a document into a zip for download."""
    if not tables:
        return None
    safe = re.sub(r"[^\w]", "_", doc_name)[:30]
    zip_path = os.path.join(WORK_DIR, f"{safe}_tables.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for t in tables:
            if os.path.exists(t["path"]):
                z.write(t["path"], os.path.basename(t["path"]))
    print(f"Zipped {len(tables)} table(s)")
    return zip_path


# ══════════════════════════════════════════════════════════════════════════════
# OCR ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def ocr_page(image: Image.Image, page_num: int, doc_name: str) -> dict:
    """OCR one page, parse any pipe tables. Returns dict with text + tables."""
    image = _normalize_image(image)
    raw = _raw_ocr(image, page_num)
    tables = parse_pipe_tables(raw, page_num, doc_name)

    if _should_retry_sparse_ocr(raw, tables):
        retry_raw = _raw_ocr(_enhance_for_retry(image), page_num)
        if _text_signal_count(retry_raw) > _text_signal_count(raw) + 20:
            raw = retry_raw
            tables = parse_pipe_tables(raw, page_num, doc_name)

    visual_summary = ""
    if _should_capture_visual_summary(doc_name, raw, tables):
        visual_summary = _visual_call(image, page_num)

    return {
        "text": raw,
        "tables": tables,
        "page": page_num,
        "visual_summary": visual_summary,
        "source": "ocr",
        "confidence": None,
    }


def stream_document_extraction(path: str, doc_name: str | None = None):
    """
    Stream document extraction events so the UI can show live routing progress.

    Events:
      document_started: {"event", "doc_name", "kind", "total_pages"}
      page_started: {"event", "page", "total_pages", "source"}
      page_complete: {"event", "page", "total_pages", "result"}
      complete: {"event", "pages", "tables", "routing", "total_pages"}
    """
    doc_name = doc_name or Path(path).name
    ext = Path(path).suffix.lower()

    if ext != ".pdf":
        images = file_to_images(path)
        total_pages = len(images)
        yield {
            "event": "document_started",
            "doc_name": doc_name,
            "kind": "image",
            "total_pages": total_pages,
        }
        for page_num, image in enumerate(images, 1):
            yield {
                "event": "page_started",
                "page": page_num,
                "total_pages": total_pages,
                "source": "ocr",
            }
            result = ocr_page(image, page_num, doc_name)
            yield {
                "event": "page_complete",
                "page": page_num,
                "total_pages": total_pages,
                "result": result,
            }
        yield {
            "event": "complete",
            "pages": [result],
            "tables": result["tables"],
            "routing": {"digital_pages": 0, "ocr_pages": 1, "native_table_pages": 0},
            "total_pages": total_pages,
        }
        return

    from pypdf import PdfReader

    reader = PdfReader(path)
    total_pages = min(len(reader.pages), MAX_PAGES)
    print(f"PDF: {total_pages} page(s) via parser-first routing")
    yield {
        "event": "document_started",
        "doc_name": doc_name,
        "kind": "pdf",
        "total_pages": total_pages,
    }

    all_pages = []
    all_tables = []
    routing = {"digital_pages": 0, "ocr_pages": 0, "native_table_pages": 0}

    for page_num in range(1, total_pages + 1):
        page = reader.pages[page_num - 1]
        layout_text = _normalize_text(page.extract_text(extraction_mode="layout") or "")
        plain_text = _normalize_text(page.extract_text(extraction_mode="plain") or "")
        native_text = _choose_native_text(layout_text, plain_text)
        native_confidence = _digital_text_confidence(native_text)
        native_tables = parse_layout_tables(layout_text, page_num, doc_name)

        if native_confidence >= 0.65 or (native_tables and _text_signal_count(native_text) >= 80):
            yield {
                "event": "page_started",
                "page": page_num,
                "total_pages": total_pages,
                "source": "digital-text",
            }
            page_tables = native_tables or parse_pipe_tables(native_text, page_num, doc_name)
            if native_tables:
                routing["native_table_pages"] += 1
            routing["digital_pages"] += 1
            page_result = {
                "text": native_text,
                "tables": page_tables,
                "page": page_num,
                "visual_summary": "",
                "source": "digital-text",
                "confidence": native_confidence,
            }
            all_pages.append(page_result)
            all_tables.extend(page_tables)
            yield {
                "event": "page_complete",
                "page": page_num,
                "total_pages": total_pages,
                "result": page_result,
            }
            continue

        yield {
            "event": "page_started",
            "page": page_num,
            "total_pages": total_pages,
            "source": "ocr",
        }
        image = _render_pdf_page(path, page_num)
        ocr_result = ocr_page(image, page_num, doc_name)
        routing["ocr_pages"] += 1
        all_pages.append(ocr_result)
        all_tables.extend(ocr_result["tables"])
        yield {
            "event": "page_complete",
            "page": page_num,
            "total_pages": total_pages,
            "result": ocr_result,
        }

    yield {
        "event": "complete",
        "pages": all_pages,
        "tables": all_tables,
        "routing": routing,
        "total_pages": total_pages,
    }


def extract_document(path: str, doc_name: str | None = None) -> dict:
    """
    Unified ingest path.

    Images go straight to OCR.
    PDFs are routed per page:
    - native text first for born-digital pages
    - OCR only for weak/scanned pages
    """
    final_result = None
    for event in stream_document_extraction(path, doc_name):
        if event["event"] == "complete":
            final_result = {
                "pages": event["pages"],
                "tables": event["tables"],
                "routing": event["routing"],
            }
    if final_result is None:
        raise RuntimeError("Document extraction completed without a final result.")
    return final_result


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def auto_summarize(combined_text: str) -> str:
    """
    Generate a ≤25-word one-sentence summary of the document.
    Used as context in the chat engine system prompt so the LLM always knows
    what it's looking at, even for queries like 'what is this document?'
    """
    snippet = combined_text[:2500].strip()
    if not snippet:
        return "Uploaded document."
    try:
        summary = _quick_call(
            SUMMARIZE_PROMPT.format(text=snippet), max_tokens=80
        )
        return summary.strip().rstrip(".")
    except Exception as e:
        print(f"  Auto-summary failed: {e}")
        return "Uploaded document."
