"""
config.py — All constants and environment configuration.
Single source of truth. Import from here everywhere else.
"""
import os

from dotenv import load_dotenv


load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR = os.environ.get("DOCQA_WORK_DIR", "/tmp/docqa_v3")
os.makedirs(WORK_DIR, exist_ok=True)

# ── API keys ──────────────────────────────────────────────────────────────────
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── App ───────────────────────────────────────────────────────────────────────
GRADIO_SHARE  = os.environ.get("GRADIO_SHARE",  "false").lower() in ("1", "true", "yes")
OPEN_BROWSER  = os.environ.get("OPEN_BROWSER",  "true").lower()  in ("1", "true", "yes")
SERVER_HOST   = os.environ.get("DOCQA_SERVER_HOST", "127.0.0.1")
SERVER_PORT   = int(os.environ.get("DOCQA_SERVER_PORT", "7860"))
API_PORT      = int(os.environ.get("DOCQA_API_PORT", str(SERVER_PORT + 1)))

# ── Document pipeline ─────────────────────────────────────────────────────────
MAX_PAGES   = 100      # hard cap per upload
DPI         = 200      # rasterisation DPI (200 = good for print + handwriting)
OCR_TOKENS  = 4096     # max tokens for OCR response
QA_TOKENS   = 4096     # max tokens for QA / summary response

# ── LlamaIndex chunking ───────────────────────────────────────────────────────
# Token-based (not word-based). SentenceSplitter uses tiktoken internally.
# 512 tokens ≈ 380 words — tight enough for precision, large enough for context.
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
TOP_K         = 8      # nodes retrieved per query

# ── Conversation ──────────────────────────────────────────────────────────────
# ChatSummaryMemoryBuffer token budget.
# 3000 tokens ≈ 20+ message turns before summarisation kicks in.
MEMORY_TOKEN_LIMIT = 3000
MSG_LIMIT          = 20   # Gradio chat messages before context clear
MSG_WARN_AT        = 18   # show warning N messages before clear

# ── Backend selection ─────────────────────────────────────────────────────────
_forced = os.environ.get("BACKEND", "").lower()
if   _forced in ("openai", "gemini"):  BACKEND = _forced
elif OPENAI_KEY and not GEMINI_KEY:    BACKEND = "openai"
elif GEMINI_KEY and not OPENAI_KEY:    BACKEND = "gemini"
elif OPENAI_KEY and GEMINI_KEY:        BACKEND = "openai"
else:
    raise SystemExit(
        "\n❌  No API key found.\n"
        "    OPENAI_API_KEY='sk-...'   python app.py\n"
        "    GEMINI_API_KEY='AIza...'  python app.py\n"
    )

OCR_MODEL     = "gpt-4.1"
QA_MODEL      = "gpt-4o-mini"
GEMINI_MODEL  = "models/gemini-2.5-flash"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"

if BACKEND == "openai":
    BACKEND_LABEL = "GPT-4.1 OCR + GPT-4o mini"
    BADGE_COLOR   = "#60a5fa"
else:
    BACKEND_LABEL = "Gemini 2.5 Flash"
    BADGE_COLOR   = "#a78bfa"
