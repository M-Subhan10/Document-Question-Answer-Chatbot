from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .backends import BaseBackend
from .config import Settings
from .prompts import load_prompt
from .types import DocumentRecord, PageRecord, TableArtifact


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"}


def detect_file_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in SUPPORTED_IMAGE_EXTS:
        return "image"
    raise ValueError(f"Unsupported file type: {ext}")


def make_doc_id(name: str, seen: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-") or "document"
    base = slug
    counter = 2
    while slug in seen:
        slug = f"{base}-{counter}"
        counter += 1
    seen.add(slug)
    return slug


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def digital_text_confidence(text: str) -> float:
    if not text.strip():
        return 0.0
    text = text.strip()
    words = text.split()
    if len(words) < 25:
        return 0.25
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, len(text))
    space_ratio = text.count(" ") / max(1, len(text))
    weird_ratio = sum(ch in {"�", "\u25a1"} for ch in text) / max(1, len(text))
    score = 0.45
    score += min(len(words) / 300, 0.3)
    score += min(alpha_ratio, 0.2)
    score += 0.1 if 0.08 <= space_ratio <= 0.25 else 0.0
    score -= weird_ratio * 2
    return max(0.0, min(score, 1.0))


def ocr_confidence(text: str) -> float:
    if not text.strip():
        return 0.0
    words = text.split()
    unclear_count = text.lower().count("[unclear]")
    unclear_ratio = unclear_count / max(1, len(words))
    line_count = max(1, len(text.splitlines()))
    average_line = len(text) / line_count
    score = 0.4
    score += min(len(words) / 400, 0.25)
    score += 0.15 if average_line > 20 else 0.0
    score -= min(unclear_ratio, 0.5)
    if "|" in text:
        score += 0.1
    return max(0.0, min(score, 1.0))


def parse_pipe_tables(
    text: str,
    doc_id: str,
    doc_name: str,
    page_num: int,
    work_dir: Path,
) -> list[TableArtifact]:
    tables: list[TableArtifact] = []
    buffer: list[list[str]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) >= 2:
            buffer.append(parts)
            continue
        if len(buffer) >= 2:
            _flush_table(buffer, doc_id, doc_name, page_num, tables, work_dir)
        buffer = []
    if len(buffer) >= 2:
        _flush_table(buffer, doc_id, doc_name, page_num, tables, work_dir)
    return tables


def _flush_table(
    rows: list[list[str]],
    doc_id: str,
    doc_name: str,
    page_num: int,
    tables: list[TableArtifact],
    work_dir: Path,
) -> None:
    headers, data = rows[0], rows[1:]
    cleaned_rows = []
    for row in data:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[: len(headers)]
        cleaned_rows.append(row)
    if not cleaned_rows:
        return
    df = pd.DataFrame(cleaned_rows, columns=headers)
    path = work_dir / f"{doc_id}_p{page_num}_table_{len(tables)}.csv"
    df.to_csv(path, index=False)
    tables.append(
        TableArtifact(
            doc_id=doc_id,
            doc_name=doc_name,
            page_num=page_num,
            path=path,
            rows=len(df),
            cols=list(df.columns),
        )
    )


class DocumentExtractor:
    def __init__(self, settings: Settings, primary_backend: BaseBackend, backup_backend: BaseBackend | None):
        self.settings = settings
        self.primary_backend = primary_backend
        self.backup_backend = backup_backend
        self.ocr_prompt = load_prompt("ocr.txt")

    def extract(self, doc_id: str, path: Path, remaining_page_budget: int) -> DocumentRecord:
        kind = detect_file_kind(path)
        if kind == "pdf":
            return self._extract_pdf(doc_id, path, remaining_page_budget)
        return self._extract_image(doc_id, path)

    def _extract_pdf(self, doc_id: str, path: Path, remaining_page_budget: int) -> DocumentRecord:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        total_pages = min(len(reader.pages), self.settings.max_pages_per_file, remaining_page_budget)
        warnings: list[str] = []
        if total_pages <= 0:
            raise ValueError("No page budget left for this file.")
        if len(reader.pages) > total_pages:
            warnings.append(
                f"Page cap applied. Processed {total_pages} of {len(reader.pages)} page(s)."
            )

        pages: list[PageRecord] = []
        for page_num in range(1, total_pages + 1):
            extracted = normalize_text(reader.pages[page_num - 1].extract_text() or "")
            confidence = digital_text_confidence(extracted)
            if confidence >= 0.65:
                tables = parse_pipe_tables(extracted, doc_id, path.name, page_num, self.settings.work_dir)
                pages.append(
                    PageRecord(
                        doc_id=doc_id,
                        doc_name=path.name,
                        page_num=page_num,
                        text=extracted,
                        source="digital-text",
                        confidence=confidence,
                        tables=tables,
                    )
                )
                continue

            image = self._render_pdf_page(path, page_num)
            pages.append(self._ocr_page(doc_id, path.name, page_num, image, fallback_hint="digital-text extraction too weak"))

        return DocumentRecord(
            doc_id=doc_id,
            name=path.name,
            kind="pdf",
            path=path,
            page_count=len(pages),
            pages=pages,
            summary="",
            warnings=warnings,
        )

    def _extract_image(self, doc_id: str, path: Path) -> DocumentRecord:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        page = self._ocr_page(doc_id, path.name, 1, image, fallback_hint="single-image document")
        return DocumentRecord(
            doc_id=doc_id,
            name=path.name,
            kind="image",
            path=path,
            page_count=1,
            pages=[page],
            summary="",
            warnings=[],
        )

    def _render_pdf_page(self, path: Path, page_num: int):
        from pdf2image import convert_from_path

        pages = convert_from_path(
            str(path),
            dpi=self.settings.dpi,
            first_page=page_num,
            last_page=page_num,
            fmt="PNG",
            thread_count=1,
        )
        return pages[0]

    def _ocr_page(self, doc_id: str, doc_name: str, page_num: int, image, fallback_hint: str) -> PageRecord:
        warnings: list[str] = []
        primary_text = self.primary_backend.ocr_image(image, self.ocr_prompt)
        best_text = normalize_text(primary_text)
        best_score = ocr_confidence(best_text)
        source = "ocr-primary"

        if best_score < 0.52:
            warnings.append(f"Low OCR confidence on page {page_num} ({fallback_hint}).")
            if self.backup_backend is not None:
                backup_text = normalize_text(self.backup_backend.ocr_image(image, self.ocr_prompt))
                backup_score = ocr_confidence(backup_text)
                if backup_score > best_score:
                    best_text = backup_text
                    best_score = backup_score
                    source = "ocr-backup"
                    warnings.append(f"Backup OCR backend selected for page {page_num}.")

        tables = parse_pipe_tables(best_text, doc_id, doc_name, page_num, self.settings.work_dir)
        return PageRecord(
            doc_id=doc_id,
            doc_name=doc_name,
            page_num=page_num,
            text=best_text,
            source=source,
            confidence=best_score,
            warnings=warnings,
            tables=tables,
        )
