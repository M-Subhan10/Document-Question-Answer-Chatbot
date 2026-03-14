from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    work_dir: Path = Path(os.environ.get("WORK_DIR", "/tmp/handwritten_qa"))
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    forced_backend: str = os.environ.get("BACKEND", "").strip().lower()
    gradio_share: bool = os.environ.get("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")
    open_browser: bool = os.environ.get("OPEN_BROWSER", "true").lower() in ("1", "true", "yes")
    server_host: str = os.environ.get("DOCQA_SERVER_HOST", "127.0.0.1")
    server_port: int = int(os.environ.get("DOCQA_SERVER_PORT", "7860"))
    dpi: int = int(os.environ.get("DOCQA_DPI", "180"))
    max_files: int = int(os.environ.get("DOCQA_MAX_FILES", "3"))
    max_pages_per_file: int = int(os.environ.get("DOCQA_MAX_PAGES_PER_FILE", "40"))
    max_total_pages: int = int(os.environ.get("DOCQA_MAX_TOTAL_PAGES", "60"))
    max_file_size_mb: int = int(os.environ.get("DOCQA_MAX_FILE_SIZE_MB", "40"))
    chunk_words: int = int(os.environ.get("DOCQA_CHUNK_WORDS", "240"))
    overlap_words: int = int(os.environ.get("DOCQA_OVERLAP_WORDS", "60"))
    retrieval_k: int = int(os.environ.get("DOCQA_RETRIEVAL_K", "10"))
    warn_after_turns: int = int(os.environ.get("DOCQA_WARN_AFTER_TURNS", "16"))
    rollover_after_turns: int = int(os.environ.get("DOCQA_ROLLOVER_AFTER_TURNS", "20"))
    embed_model: str = os.environ.get("DOCQA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    openai_ocr_model: str = os.environ.get("OPENAI_OCR_MODEL", "gpt-4.1")
    openai_chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    gemini_ocr_model: str = os.environ.get("GEMINI_OCR_MODEL", "gemini-2.5-flash")
    gemini_chat_model: str = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    history_file: Path = field(default_factory=lambda: Path(os.environ.get("DOCQA_HISTORY_FILE", "/tmp/handwritten_qa/chat_history.json")))

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)
