# app.py — Dual-backend Document Q&A  ·  Chatbot Edition
# ──────────────────────────────────────────────────────
# Run with Gemini:  GEMINI_API_KEY='AIza...'  python app.py
# Run with OpenAI:  OPENAI_API_KEY='sk-...'   python app.py
# Both set?         BACKEND=openai python app.py   (or BACKEND=gemini)
#
# ── Speed note ──────────────────────────────────────────────────────────────
#  GPT-4o-mini  ≈ 400–900 ms first token  (fastest cheap OpenAI model)
#  GPT-4o       ≈ 300–600 ms first token  (better quality, ~3× cost)
#  Gemini 2.5 Flash ≈ 150–350 ms          (Google's infra is genuinely faster)
#  We use streaming so responses feel instant regardless of first-token latency.
#  Swap QA_MODEL='gpt-4o' below if you want higher quality at ~3× cost.
# ────────────────────────────────────────────────────────────────────────────
import os, re, time, base64, mimetypes, zipfile, io, traceback, requests
from pathlib import Path
from html import escape as _esc

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_KEY   = os.environ.get('GEMINI_API_KEY', '')
OPENAI_KEY   = os.environ.get('OPENAI_API_KEY', '')
WORK_DIR     = '/tmp/handwritten_qa'
GRADIO_SHARE = os.environ.get('GRADIO_SHARE', 'false').lower() in ('1','true','yes')
OPEN_BROWSER = os.environ.get('OPEN_BROWSER', 'true').lower()  in ('1','true','yes')
MAX_PAGES    = 100
DPI          = 350

# ── Backend auto-detection ────────────────────────────────────────────────────
_forced = os.environ.get('BACKEND', '').lower()
if   _forced in ('openai', 'gemini'):   BACKEND = _forced
elif OPENAI_KEY and not GEMINI_KEY:     BACKEND = 'openai'
elif GEMINI_KEY and not OPENAI_KEY:     BACKEND = 'gemini'
elif OPENAI_KEY and GEMINI_KEY:         BACKEND = 'openai'
else:
    raise SystemExit(
        '\n❌  No API key found.\n'
        '    Set GEMINI_API_KEY  or  OPENAI_API_KEY before launching.\n\n'
        '    Gemini:  GEMINI_API_KEY="AIza..."  python app.py\n'
        '    OpenAI:  OPENAI_API_KEY="sk-..."   python app.py\n'
    )

os.makedirs(WORK_DIR, exist_ok=True)
print(f'Backend selected: {BACKEND.upper()}')


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 — PDF / Image → PIL  (shared, no API)
# ══════════════════════════════════════════════════════════════════════════════
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_images(path):
    pages = convert_from_path(path, dpi=DPI, first_page=1, last_page=MAX_PAGES,
                              fmt='JPEG', thread_count=2)
    print(f'PDF: {len(pages)} page(s) at {DPI} DPI');  return pages

def load_image_file(path):
    img = Image.open(path).convert('RGB');  print(f'Image: {Path(path).name}');  return [img]

def file_to_images(path):
    ext = Path(path).suffix.lower()
    if ext == '.pdf':  return pdf_to_images(path)
    if ext in ('.png','.jpg','.jpeg','.tiff','.tif','.webp','.bmp'): return load_image_file(path)
    raise ValueError(f'Unsupported type: {ext}')

print('Cell 4 ready')


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — OCR engine  (Gemini Vision  OR  OpenAI GPT-4o-mini Vision)
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd

OCR_PROMPT = """You are a document digitisation expert. This image shows one page from a document.
The document may be printed (CV, report, form, letter, invoice) OR handwritten (register, form, diary).
It may contain English, Urdu, Roman-Urdu, or a mix.

Your job: extract EVERY piece of text visible on this page, completely and accurately.

━━━ EXTRACTION RULES ━━━

1. TRANSCRIBE EVERYTHING visible on the page:
   - Headings, titles, section names
   - Body paragraphs and bullet points
   - Names, job titles, organisations, places
   - Dates, phone numbers, email addresses, URLs
   - Numbers, IDs, codes, serial numbers
   - Captions, footnotes, headers, footers
   - Any handwritten annotations on printed pages
   - Content in ALL columns — do NOT stop at centre-page

2. TABLES & GRIDS: If the page contains rows and columns (register, spreadsheet,
   invoice, skills table, work history table, etc.) format each row as:
   Column1 | Column2 | Column3 | ...
   Value1  | Value2  | Value3  | ...
   Rules:
   • Always output the HEADER row first
   • Keep ALL columns — especially rightmost ones (Signature, Kind of Animal, Status, etc.)
   • Every data row must have the same pipe count as the header row
   • Use a blank cell ("") for empty cells, not skip them

3. PRINTED TEXT: For clean printed documents (CVs, reports, contracts):
   - Preserve the logical reading order (column-by-column for multi-column layouts)
   - Keep bullet points and sub-bullets; prefix bullets with "• "
   - Preserve bold/section labels as they appear (e.g. "Experience:", "Skills:")
   - Do NOT skip any section, even if it seems minor

4. HANDWRITTEN TEXT: For handwritten content:
   - Write [unclear] for any word you cannot confidently read
   - Do NOT guess or invent content

5. LANGUAGE: Do not translate. Transcribe in the original language and script.

6. STRUCTURE MARKERS (use when applicable):
   [HEADER]  — document title or top metadata block
   [TABLE]   — precedes a table / grid
   [SECTION] — major section heading (e.g. Work Experience, Education)
   [NOTES]   — footer, margin annotations, or end-of-page notes

━━━ OUTPUT ━━━
Output the full transcription now. Cover the entire page from top-left to bottom-right.
Do not summarise, skip, or truncate anything."""


def _parse_pipe_tables(text, page_num):
    """
    TABLE HANDLING OVERVIEW
    ───────────────────────
    The OCR prompt instructs the model to output any tabular data
    (registers, forms with rows/columns) as pipe-separated values, e.g.:

        Name | Phone | Address | Animal
        M. Abid | 0333-3796280 | Gile 288/HR | Calf

    This function:
      1. Scans every line of the OCR output
      2. Collects consecutive lines that have ≥2 pipe-separated parts
         (was ≥3 — changed so rightmost columns like "Kind of Animal"
          are not silently dropped when OCR misses the last pipe)
      3. When the pipe-table block ends, calls _flush_table() which:
           - Treats row[0] as column headers
           - Pads/truncates subsequent rows to match header width
           - Saves a CSV to WORK_DIR as  table_p{page}_{index}.csv
           - Appends metadata {page, path, rows, cols} to the returned list
      4. Returns all tables found on this page

    Later, S['tables'] accumulates these across all pages, and
    make_zip() bundles every CSV into /tmp/tables.zip for download.
    The source panel also reads these CSVs to render inline HTML tables.
    TABLE ROW CHUNKS: after all tables are saved, tables_to_row_chunks()
    converts every CSV row into a natural-language sentence that is added
    to the FAISS index alongside regular text chunks, so column-level
    queries like "what animals are on page 1" can find specific cells.
    """
    tables, buf = [], []
    for line in text.split('\n'):
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 2:                 # ← was 3; now 2 so last column survives
            buf.append(parts)
        else:
            if len(buf) >= 2: _flush_table(buf, page_num, tables)
            buf = []
    if len(buf) >= 2: _flush_table(buf, page_num, tables)
    return tables


def tables_to_row_chunks(tables):
    """
    Convert every saved CSV table into one natural-language sentence per row.

    Example output sentence:
      "[Page 1][TABLE ROW] Yearly No: 52977/52997 | Monthly No: 674/6769 |
       Name of Owner and Parentage: M. Abid s/o Mohammad Sheikh |
       Address: Gile 288/HR | Mobile No: 0333-3796280 |
       Kind of Animal: Calf"

    These sentences are self-contained and highly queryable — FAISS can
    find "Calf" under "Kind of Animal on page 1" because the page tag,
    column name, and value all appear together in one chunk.
    """
    row_chunks = []
    for t in tables:
        try:
            df = pd.read_csv(t['path'])
            for _, row in df.iterrows():
                parts = ' | '.join(
                    f'{col}: {val}'
                    for col, val in zip(df.columns, row)
                    if str(val).strip() and str(val).strip().lower() not in ('nan', '')
                )
                if parts:
                    row_chunks.append(f'[Page {t["page"]}][TABLE ROW] {parts}')
        except Exception:
            pass
    return row_chunks

def _flush_table(rows, page_num, tables):
    headers, data = rows[0], rows[1:]
    padded = []
    for r in data:
        if len(r) < len(headers):   r = r + ['']*(len(headers)-len(r))
        elif len(r) > len(headers): r = r[:len(headers)]
        padded.append(r)
    if not padded: return
    df   = pd.DataFrame(padded, columns=headers)
    path = os.path.join(WORK_DIR, f'table_p{page_num}_{len(tables)}.csv')
    df.to_csv(path, index=False)
    tables.append({'page': page_num, 'path': path, 'rows': len(df), 'cols': list(df.columns)})
    print(f'    Table: {len(df)}r × {len(df.columns)}c → {os.path.basename(path)}')

def _img_to_b64(img, quality=85):
    buf = io.BytesIO();  img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ── Gemini backend ────────────────────────────────────────────────────────────
if BACKEND == 'gemini':
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)
    _ocr_m        = genai.GenerativeModel('gemini-2.5-flash')
    _qa_m         = genai.GenerativeModel('gemini-2.5-flash')
    BACKEND_LABEL = 'Gemini 2.5 Flash'
    BADGE_COLOR   = '#a78bfa'

    def _raw_ocr(image, page_num):
        for attempt in range(3):
            try:
                r = _ocr_m.generate_content(
                    [OCR_PROMPT, image],
                    generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=4096))
                return r.text.strip()
            except Exception as e:
                err = str(e)
                if '429' in err or 'quota' in err.lower() or 'rate' in err.lower():
                    w = 15*(attempt+1);  print(f'  Rate limit p{page_num}, wait {w}s');  time.sleep(w)
                else:
                    print(f'  OCR err p{page_num} attempt {attempt+1}: {err[:80]}');  time.sleep(3)
        return f'[OCR failed page {page_num}]'

    def _raw_qa(prompt):
        r = _qa_m.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=4096))
        return r.text.strip()

    def _stream_qa(prompt):
        """Gemini: yield chunks. Try streaming first, fall back to single chunk."""
        try:
            resp = _qa_m.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=4096),
                stream=True,
            )
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        except Exception:
            yield _raw_qa(prompt)

    print(f'Cell 5 ready: Gemini OCR ({BACKEND_LABEL})')


# ── OpenAI backend ────────────────────────────────────────────────────────────
else:
    from openai import OpenAI
    _oa           = OpenAI(api_key=OPENAI_KEY)
    OCR_MODEL     = 'gpt-4o-mini'   # Vision OCR  — fast & cheap
    QA_MODEL      = 'gpt-4o-mini'   # Chat QA     — swap to 'gpt-4o' for higher quality
    BACKEND_LABEL = 'GPT-4o mini'
    BADGE_COLOR   = '#60a5fa'

    def _raw_ocr(image, page_num):
        b64 = _img_to_b64(image)
        for attempt in range(3):
            try:
                r = _oa.chat.completions.create(
                    model=OCR_MODEL, max_tokens=4096, temperature=0.1,
                    messages=[{'role':'user','content':[
                        {'type':'text',      'text': OCR_PROMPT},
                        {'type':'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}', 'detail':'high'}},
                    ]}])
                return r.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                if '429' in err or 'rate' in err.lower():
                    w = 10*(attempt+1);  print(f'  Rate limit p{page_num}, wait {w}s');  time.sleep(w)
                else:
                    print(f'  OCR err p{page_num} attempt {attempt+1}: {err[:80]}');  time.sleep(3)
        return f'[OCR failed page {page_num}]'

    def _raw_qa(prompt):
        r = _oa.chat.completions.create(
            model=QA_MODEL, max_tokens=4096, temperature=0.2,
            messages=[{'role':'user','content': prompt}])
        return r.choices[0].message.content.strip()

    def _stream_qa(prompt):
        """Stream tokens from OpenAI — makes responses feel instant regardless of latency."""
        stream = _oa.chat.completions.create(
            model=QA_MODEL, max_tokens=4096, temperature=0.2, stream=True,
            messages=[{'role':'user','content': prompt}])
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    print(f'Cell 5 ready: OpenAI OCR ({BACKEND_LABEL})')


# ── Shared OCR entry point ────────────────────────────────────────────────────
def ocr_page(image, page_num):
    raw    = _raw_ocr(image, page_num)
    tables = _parse_pipe_tables(raw, page_num)
    return {'text': raw, 'tables': tables, 'page': page_num}


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — Chunker + FAISS  (shared, local model)
# ══════════════════════════════════════════════════════════════════════════════
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_WORDS  = 400;  OVERLAP_WORDS = 80
BGE_PREFIX   = 'Represent this sentence for searching relevant passages: '
SCORE_THRESH = 0.25   # cosine similarity floor to accept a hit

print('Loading: BAAI/bge-small-en-v1.5')
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Embedder ready (384-dim, local CPU)')

def chunk_text(text):
    words, chunks, start = text.split(), [], 0
    while start < len(words):
        end = min(start+CHUNK_WORDS, len(words))
        c   = ' '.join(words[start:end])
        if c.strip(): chunks.append(c)
        start += CHUNK_WORDS - OVERLAP_WORDS
    return chunks

def build_index(text, extra_chunks=None):
    """
    Build FAISS index from OCR text + table row sentences.

    extra_chunks: list of row-sentence strings from tables_to_row_chunks().
    They are appended AFTER the regular text chunks so their indices align.
    This gives FAISS direct access to every cell value with its column name
    and page tag, fixing queries like "what animals are on page 1" which
    previously failed because a 400-word sliding-window chunk could split
    a table mid-row, losing the column→value pairing.
    """
    chunks = chunk_text(text)
    if extra_chunks:
        chunks = chunks + extra_chunks
        print(f'  (+{len(extra_chunks)} table-row chunks)')
    if not chunks: raise ValueError('No text to index.')
    print(f'Embedding {len(chunks)} chunks…')
    vecs = embedder.encode([BGE_PREFIX+c for c in chunks], batch_size=64,
                           normalize_embeddings=True, show_progress_bar=True).astype('float32')
    idx = faiss.IndexFlatIP(vecs.shape[1]);  idx.add(vecs)
    print(f'FAISS ready: {idx.ntotal} vectors');  return idx, chunks

def search_with_scores(query, index, chunks, top_k=5):
    """Return list of (chunk_text, score) sorted by relevance."""
    qv = embedder.encode([BGE_PREFIX+query], normalize_embeddings=True).astype('float32')
    scores, ids = index.search(qv, top_k)
    return [(chunks[i], float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0 and s > SCORE_THRESH]

print('Cell 6 ready')


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — Audio transcription  (Whisper-1  OR  Gemini audio)
# ══════════════════════════════════════════════════════════════════════════════
def transcribe(audio_path):
    if not audio_path or not os.path.exists(audio_path): return ''
    try:
        if BACKEND == 'openai':
            with open(audio_path, 'rb') as f:
                r = _oa.audio.transcriptions.create(model='whisper-1', file=f, response_format='text')
            text = (r if isinstance(r, str) else r.text).strip()
        else:
            ab   = Path(audio_path).read_bytes()
            if len(ab) > 20*1024*1024: print('Audio too large'); return ''
            mime = mimetypes.guess_type(audio_path)[0] or 'audio/wav'
            payload = {'contents':[{'parts':[
                {'text':'Return only the spoken words, no timestamps or labels.'},
                {'inline_data':{'mime_type': mime, 'data': base64.b64encode(ab).decode()}},
            ]}]}
            resp = requests.post(
                'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent',
                headers={'Content-Type':'application/json','x-goog-api-key':GEMINI_KEY},
                json=payload, timeout=120)
            resp.raise_for_status()
            parts = resp.json().get('candidates',[{}])[0].get('content',{}).get('parts',[])
            text  = ' '.join(p.get('text','') for p in parts).strip()
    except Exception as e:
        print(f'STT error: {e}');  return ''
    print(f'Transcribed: "{text[:80]}"');  return text

print('Cell 7 ready')


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Table zip  (shared)
# ══════════════════════════════════════════════════════════════════════════════
def make_zip(tables):
    if not tables: return None
    zp = '/tmp/tables.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for t in tables:
            if os.path.exists(t['path']): zf.write(t['path'], arcname=os.path.basename(t['path']))
    print(f'Zipped {len(tables)} table(s)');  return zp

print('Cell 8 ready')


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8b — Chat History Persistence
# ══════════════════════════════════════════════════════════════════════════════
import json, datetime

HISTORY_FILE = os.path.join(WORK_DIR, 'chat_history.json')

def _load_history():
    """Load all past sessions from disk. Returns list of session dicts."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_turn(doc_name: str, question: str, answer: str, intent: str):
    """Append a single Q&A turn to the history file."""
    sessions = _load_history()
    # Find today's session for this doc
    today = datetime.date.today().isoformat()
    key   = f'{doc_name}::{today}'
    # Find existing session or create new
    session = None
    for s in sessions:
        if s.get('key') == key:
            session = s
            break
    if session is None:
        session = {
            'key':    key,
            'doc':    doc_name or '(no document)',
            'date':   today,
            'turns':  [],
        }
        sessions.append(session)
    session['turns'].append({
        'q':      question,
        'a':      answer,
        'intent': intent,
        'time':   datetime.datetime.now().strftime('%H:%M'),
    })
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'History save error: {e}')

def _render_history_html():
    """Render all past sessions as styled read-only HTML cards."""
    sessions = _load_history()
    if not sessions:
        return (
            '<div style="padding:12px 4px;color:#374151;font-family:monospace;'
            'font-size:11px;text-align:center;line-height:1.8">'
            'No history yet.<br>Conversations will appear here.</div>'
        )

    # Newest first
    sessions = list(reversed(sessions))
    parts = []
    intent_ico = {'document':'📄','general':'🧠','web':'🌐','chitchat':'💬'}

    for si, sess in enumerate(sessions[:20]):   # cap at 20 sessions shown
        turns  = sess.get('turns', [])
        doc    = _esc(sess.get('doc', '?'))
        date   = sess.get('date', '')
        n      = len(turns)
        sid    = f'hs_{si}'

        # Build inner turns HTML
        turns_html = ''
        for t in turns:
            ico = intent_ico.get(t.get('intent',''), '·')
            q   = _esc(t.get('q','')[:120])
            a   = _esc(t.get('a','')[:200])
            tm  = _esc(t.get('time',''))
            turns_html += f'''
            <div style="padding:8px 0;border-bottom:1px solid #1a1d29">
              <div style="display:flex;gap:6px;align-items:baseline;margin-bottom:3px">
                <span style="font-size:9px;color:#374151;font-family:monospace">{tm}</span>
                <span style="font-size:9px;color:#5a6278">{ico}</span>
                <span style="font-size:12px;color:#dde1ec;font-weight:500">{q}{"…" if len(t.get("q",""))>120 else ""}</span>
              </div>
              <div style="font-size:11.5px;color:#5a6278;line-height:1.6;padding-left:2px">{a}{"…" if len(t.get("a",""))>200 else ""}</div>
            </div>'''

        parts.append(f'''
        <details style="margin-bottom:8px" {"open" if si==0 else ""}>
          <summary style="cursor:pointer;list-style:none;padding:8px 10px;
                          background:#131520;border:1px solid #23273a;border-radius:7px;
                          display:flex;align-items:center;gap:8px;user-select:none">
            <span style="font-size:10px;font-family:monospace;color:#f0a500;letter-spacing:.05em">▶</span>
            <span style="flex:1;font-size:12px;color:#dde1ec;font-family:monospace;
                         white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{doc}</span>
            <span style="font-size:10px;color:#374151;font-family:monospace;white-space:nowrap">{date} · {n}Q</span>
          </summary>
          <div style="border:1px solid #1e2130;border-top:none;border-radius:0 0 7px 7px;
                      padding:4px 10px 4px;background:#0c0e14">
            {turns_html}
          </div>
        </details>''')

    return ''.join(parts)


print('Cell 8b ready: history persistence')


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Chatbot helpers
# ══════════════════════════════════════════════════════════════════════════════
import gradio as gr

# ── App state ─────────────────────────────────────────────────────────────────
S = {'index':None,'chunks':None,'tables':[],'zip':None,'name':'','pages':0}

def _ts(): return time.strftime('%H:%M:%S')

# ── Animated "thinking" strings shown while LLM is processing ────────────────
THINKING_MSGS = [
    "🤔 Analysing your question…",
    "🔍 Scanning document chunks…",
    "📊 Checking extracted tables…",
    "🧩 Cross-referencing passages…",
    "📐 Scoring relevance vectors…",
    "✍️ Composing answer…",
]


# ── Synthesis / table intent detection ────────────────────────────────────────
_TABLE_KEYWORDS = {
    'table','list','all','every','each','show','display','create','make',
    'give','provide','summarize','summarise','summary','overview','compile',
    'animals','animal','owners','owner','diagnosis','diagnoses',
    'names','addresses','address','phones','phone','numbers','mobile',
    'entries','records','rows','data','details','information','info',
    'columns','fields','register','combine','join','merge','cross',
    'compare','comparison','total','count','how many',
}

def _is_synthesis_query(q: str) -> bool:
    """
    Returns True when the query is asking for a table, list, summary,
    or any synthesis that requires looking at ALL data across pages,
    not just the top FAISS hits.

    Examples that should match:
      - "Make a table of owners and their animals"
      - "List all animals"
      - "Show me all the data"
      - "Create a summary table"
      - "Tell me about the animals"
      - "Give me all entries"
    """
    q_lower = q.lower().strip()
    words = set(re.split(r'\W+', q_lower))
    # Direct table/list creation keywords
    action_words = words & {'table','list','create','make','show','display',
                            'give','provide','summarize','summarise','summary',
                            'compile','overview','combine','merge','all','every','each'}
    data_words   = words & {'animals','animal','owners','owner','diagnosis',
                            'diagnoses','names','addresses','phone','phones',
                            'entries','records','rows','data','details',
                            'information','register','columns','fields','mobile'}
    # If they mention a table action OR if they want "all" of something
    if action_words and data_words:
        return True
    # "make a table", "create a list", "show all data"
    if re.search(r'\b(make|create|build|generate|give|show|display)\b.*\b(table|list|summary|overview)\b', q_lower):
        return True
    if re.search(r'\b(table|list|summary)\b.*\b(of|for|with|about|from)\b', q_lower):
        return True
    # "all animals", "every owner", "list of"
    if re.search(r'\b(all|every|each|list of|tell me about the)\b', q_lower) and data_words:
        return True
    # "how many" is synthesis
    if 'how many' in q_lower:
        return True
    return False


# ── Intent detection ──────────────────────────────────────────────────────────
def _detect_intent(hits_with_scores, question=''):
    """
    INTENT DETECTION
    ────────────────
    'chitchat'  → greeting / small-talk detected before FAISS  (bypass everything)
    'document'  → best FAISS hit ≥ threshold  OR synthesis query with doc loaded
    'general'   → doc loaded but no strong match  →  LLM general knowledge
    'web'       → no doc loaded  →  DuckDuckGo + LLM
    """
    if S['index'] is None:
        return 'web'

    # Synthesis queries always get document intent when tables exist
    if _is_synthesis_query(question) and S.get('tables'):
        return 'document'

    if not hits_with_scores:
        return 'general'
    best_score = hits_with_scores[0][1]
    if best_score >= 0.48:
        return 'document'
    # Lower threshold for queries that at least mention data-related terms
    if best_score >= 0.35:
        q_lower = question.lower()
        data_terms = {'animal','owner','name','phone','address','diagnosis',
                      'mobile','daily','monthly','yearly','kind','type'}
        if any(t in q_lower for t in data_terms):
            return 'document'
    return 'general'


# ── Chitchat / greeting pre-filter ────────────────────────────────────────────
_CHITCHAT_TOKENS = {
    'hello','hi','hey','howdy','hiya','sup','wassup',
    'how are you','how are u','how r u','how ru','how r you',
    'good morning','good afternoon','good evening','good night',
    'thanks','thank you','thank u','thx','ty',
    'bye','goodbye','see you','see ya','cya',
    'who are you','what are you','what can you do','what do you do',
    'are you an ai','are you a bot','are you human',
    'help','what is this','how does this work',
    'ok','okay','cool','nice','great','awesome','sure','yep','nope','no','yes',
}

def _is_chitchat(q: str) -> bool:
    """
    Returns True when the question is small-talk / greeting with no document intent.

    Checks:
      1. Exact or near-exact match against known chitchat phrases
      2. Very short queries (≤ 3 words) that don't contain document keywords
    """
    q_lower = q.lower().strip().rstrip('?!.,').strip()   # ← extra .strip() kills trailing space left by rstrip
    if q_lower in _CHITCHAT_TOKENS:
        return True
    for token in _CHITCHAT_TOKENS:
        if q_lower == token or q_lower.startswith(token + ' ') or q_lower.endswith(' ' + token):
            if len(q.split()) <= 6:
                return True
    # Very short with no document-like words
    doc_words = {'page','document','table','name','address','phone','mobile',
                 'animal','date','number','column','row','register','list',
                 'what','who','when','where','which','how many','tell me'}
    words = set(q_lower.split())
    if len(words) <= 3 and not (words & doc_words):
        return True
    return False


# ── DuckDuckGo instant-answer (no API key) ────────────────────────────────────
def _ddg_search(query, max_r=3):
    try:
        r = requests.get('https://api.duckduckgo.com/',
                         params={'q':query,'format':'json','no_html':1,'skip_disambig':1},
                         timeout=6)
        d = r.json()
        results = []
        if d.get('Abstract'):
            results.append(d['Abstract'])
        for t in d.get('RelatedTopics', [])[:max_r]:
            if isinstance(t, dict) and t.get('Text'):
                results.append(t['Text'])
        return results
    except:
        return []


# ── Source panel HTML ─────────────────────────────────────────────────────────
_TABLE_CSS = """
<style>
.dtable{border-collapse:collapse;width:100%;font-size:12px;
        font-family:'JetBrains Mono',monospace;margin:0}
.dtable th{background:#1e2130;color:#f0a500;padding:5px 10px;
           border:1px solid #23273a;text-align:left;font-weight:600;
           white-space:nowrap}
.dtable td{background:#13151c;color:#dde1ec;padding:4px 10px;
           border:1px solid #1e2130;white-space:nowrap}
.dtable tr:hover td{background:#191c25}
</style>"""

def _df_to_html(df):
    ths = ''.join(f'<th>{_esc(str(c))}</th>' for c in df.columns)
    body = ''
    for _, row in df.iterrows():
        tds = ''.join(f'<td>{_esc(str(v))}</td>' for v in row)
        body += f'<tr>{tds}</tr>'
    return f'<table class="dtable"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'

def _format_sources_html(hits_with_scores, tables, intent, web_snippets=None):
    """Render retrieved passages + matched CSV tables as styled HTML."""
    parts = [_TABLE_CSS]

    # ── Intent badge ─────────────────────────────────────────────────────────
    badge_map = {
        'document': ('📄', '#22c55e', 'Answered from document'),
        'general':  ('🧠', '#60a5fa', 'General LLM knowledge'),
        'web':      ('🌐', '#f0a500', 'Web + LLM knowledge'),
        'chitchat': ('💬', '#9ca3af', 'Casual chat'),
    }
    ico, col, label = badge_map.get(intent, ('📄','#9ca3af','Unknown'))
    parts.append(f'<div style="margin-bottom:14px;padding:6px 12px;'
                 f'border:1px solid {col}40;border-radius:6px;background:{col}10;'
                 f'font-family:monospace;font-size:12px;color:{col}">'
                 f'{ico} {label}</div>')

    # ── Web snippets ──────────────────────────────────────────────────────────
    if web_snippets:
        for i, snip in enumerate(web_snippets[:3], 1):
            parts.append(f'''
            <div style="margin-bottom:10px;border:1px solid #23273a;border-radius:8px;overflow:hidden">
              <div style="background:#131520;padding:5px 12px;font-family:monospace;font-size:11px;color:#f0a500">
                🌐 Web result {i}
              </div>
              <div style="padding:8px 12px;font-size:12.5px;line-height:1.65;color:#9ca3af;background:#13151c">
                {_esc(snip[:400])}
              </div>
            </div>''')

    # ── Document passages + tables ────────────────────────────────────────────
    shown_tables = set()

    if hits_with_scores:
        for i, (chunk, score) in enumerate(hits_with_scores, 1):
            pm  = re.search(r'\[Page (\d+)\]', chunk)
            pg  = int(pm.group(1)) if pm else None
            clean = re.sub(r'\[Page \d+\]', '', chunk)
            clean = clean.replace('[HEADER]','').replace('[TABLE]','').replace('[NOTES]','').strip()
            pct   = int(min(score, 1.0) * 100)
            bar_w = pct

            parts.append(f'''
            <div style="margin-bottom:14px;border:1px solid #23273a;border-radius:8px;overflow:hidden">
              <div style="background:#131520;padding:5px 12px;display:flex;align-items:center;gap:10px">
                <span style="font-family:monospace;font-size:11px;color:#5a6278">
                  📄 Passage {i}{f" · Page {pg}" if pg else ""}
                </span>
                <span style="flex:1;height:3px;background:#1e2130;border-radius:3px;overflow:hidden">
                  <span style="display:block;height:100%;width:{bar_w}%;background:#f0a500;border-radius:3px"></span>
                </span>
                <span style="font-family:monospace;font-size:10px;color:#f0a500">{pct}%</span>
              </div>
              <div style="padding:8px 12px;font-size:12px;line-height:1.75;color:#9ca3af;
                          background:#13151c;white-space:pre-wrap;max-height:110px;overflow-y:auto">{_esc(clean[:500])}{"…" if len(clean)>500 else ""}</div>
            </div>''')

            # Render tables for this page inline
            if pg is not None:
                for t in tables:
                    if t['page'] == pg and t['path'] not in shown_tables:
                        shown_tables.add(t['path'])
                        try:
                            df = pd.read_csv(t['path'])
                            parts.append(f'''
                            <div style="margin-bottom:14px;border:1px solid #f0a50060;border-radius:8px;overflow:hidden">
                              <div style="background:#131520;padding:5px 12px;
                                          font-family:monospace;font-size:11px;color:#f0a500">
                                📊 Table · Page {pg} · {len(df)} rows × {len(df.columns)} cols
                              </div>
                              <div style="padding:8px;overflow-x:auto;background:#13151c">
                                {_df_to_html(df)}
                              </div>
                            </div>''')
                        except Exception:
                            pass

    # ── Show ALL remaining tables not yet shown (for synthesis queries) ────
    for t in tables:
        if t['path'] not in shown_tables:
            shown_tables.add(t['path'])
            try:
                df = pd.read_csv(t['path'])
                parts.append(f'''
                <div style="margin-bottom:14px;border:1px solid #f0a50060;border-radius:8px;overflow:hidden">
                  <div style="background:#131520;padding:5px 12px;
                              font-family:monospace;font-size:11px;color:#f0a500">
                    📊 Table · Page {t['page']} · {len(df)} rows × {len(df.columns)} cols
                  </div>
                  <div style="padding:8px;overflow-x:auto;background:#13151c">
                    {_df_to_html(df)}
                  </div>
                </div>''')
            except Exception:
                pass

    if not hits_with_scores and not shown_tables:
        if not web_snippets:
            parts.append('<div style="color:#5a6278;font-size:12px;font-family:monospace;padding:8px">No document passages retrieved.</div>')

    return ''.join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — Upload & Q&A logic
# ══════════════════════════════════════════════════════════════════════════════

def _log(lines, pct, stage, tone='run'):
    bar  = {'run':'#f0a500','ok':'#22c55e','err':'#ef4444'}.get(tone,'#f0a500')
    anim = 'animation:sh 1.4s linear infinite;background-size:300% 100%;' if tone=='run' else ''
    rows = ''
    for kind, txt in lines[-16:]:
        c = {'ok':'#22c55e','run':'#f0a500','err':'#ef4444','dim':'#374151','inf':'#60a5fa'}.get(kind,'#9ca3af')
        rows += f'<div style="padding:1px 0;color:{c};line-height:1.75">{txt}</div>'
    return (f'<div style="border:1px solid #1e2130;border-radius:8px;overflow:hidden;'
            f'background:#0c0e14;font-family:\'JetBrains Mono\',\'Fira Mono\',monospace;font-size:12px">'
            f'<div style="display:flex;align-items:center;gap:6px;padding:7px 12px;'
            f'background:#131520;border-bottom:1px solid #1e2130">'
            f'<span style="width:9px;height:9px;border-radius:50%;background:#ff5f57;display:inline-block"></span>'
            f'<span style="width:9px;height:9px;border-radius:50%;background:#febc2e;display:inline-block"></span>'
            f'<span style="width:9px;height:9px;border-radius:50%;background:#28c840;display:inline-block"></span>'
            f'<span style="margin-left:8px;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#374151">'
            f'{_esc(stage)}</span></div>'
            f'<div style="padding:10px 14px;min-height:64px;max-height:220px;overflow-y:auto">{rows}</div>'
            f'<div style="padding:7px 14px 11px;background:#131520;border-top:1px solid #1e2130">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;'
            f'font-size:10px;letter-spacing:.1em;text-transform:uppercase">'
            f'<span style="color:#374151">Progress</span><span style="color:{bar}">{pct}%</span></div>'
            f'<div style="height:3px;background:#1e2130;border-radius:3px;overflow:hidden">'
            f'<div style="height:100%;width:{pct}%;background:{bar};border-radius:3px;'
            f'transition:width .3s ease;{anim}"></div></div></div></div>'
            f'<style>@keyframes sh{{0%{{background-position:100% 0}}100%{{background-position:-100% 0}}}}</style>')


def on_upload(file):
    """Generator: process document, yield status updates."""
    def emit(L, pct, stage, tone='run', dl=gr.update(visible=False), qa=False):
        return (gr.update(value=_log(L,pct,stage,tone), visible=True),
                dl,
                gr.update(visible=qa),
                gr.update(visible=not qa),
                [],           # reset chatbot
                '')           # reset sources

    L = []
    if file is None:
        L.append(('dim','Waiting for a file…'));  yield emit(L,0,'idle');  return

    try:
        fname = os.path.basename(file.name)
        ext   = Path(fname).suffix.upper().lstrip('.')

        L.append(('inf', f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'File: <b style="color:#e2e4eb">{_esc(fname)}</b>'))
        yield emit(L, 3, 'decode')

        L.append(('run', f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'Decoding {ext} → raster pages at {DPI} DPI…'))
        yield emit(L, 6, 'decode')

        images = file_to_images(file.name)
        n = len(images)

        L.append(('ok',  f'<span style="color:#6b7280">{_ts()}</span>  ✓ {n} page(s) decoded'))
        L.append(('dim', f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'<span style="color:#374151">Est. ~{max(1,n*3)}s  ({BACKEND_LABEL})</span>'))
        yield emit(L, 9, 'read')

        all_text, all_tables = [], []
        for i, img in enumerate(images, 1):
            p0 = int(10+(i-1)/n*58)
            L.append(('run', f'<span style="color:#6b7280">{_ts()}</span>  '
                             f'Reading page <b style="color:#e2e4eb">{i}/{n}</b>…'))
            yield emit(L, p0, f'read  {i}/{n}')

            r = ocr_page(img, i)
            all_text.append(f'[Page {i}]\n{r["text"]}')
            all_tables.extend(r['tables'])

            tl = (f'  <span style="color:#f0a500">{len(r["tables"])} tbl</span>'
                  if r['tables'] else '')
            L.append(('ok', f'<span style="color:#6b7280">{_ts()}</span>  '
                            f'✓ Page {i}  <span style="color:#374151">·</span>  '
                            f'<b style="color:#e2e4eb">{len(r["text"]):,} chars</b>{tl}'))
            yield emit(L, int(10+i/n*58), f'read  {i}/{n}')
            if i < n: time.sleep(0.5 if BACKEND=='openai' else 1.0)

        combined = '\n\n'.join(all_text)
        if not combined.strip():
            L.append(('err', f'<span style="color:#6b7280">{_ts()}</span>  '
                             '✗ No text found — check scan quality'))
            yield emit(L, 0, 'error', tone='err');  return

        L.append(('dim', f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'<span style="color:#374151">Done · {len(combined):,} chars · {len(all_tables)} table(s)</span>'))
        L.append(('run', f'<span style="color:#6b7280">{_ts()}</span>  Computing sentence embeddings…'))
        yield emit(L, 72, 'embed')

        row_chunks = tables_to_row_chunks(all_tables)
        index, chunks = build_index(combined, extra_chunks=row_chunks)
        L.append(('ok',  f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'✓ {len(chunks)} chunks  <span style="color:#374151">(384-dim · local)</span>'))
        L.append(('run', f'<span style="color:#6b7280">{_ts()}</span>  Building FAISS index…'))
        yield emit(L, 88, 'index')

        zp = make_zip(all_tables)
        S.update(index=index, chunks=chunks, tables=all_tables, zip=zp, name=fname, pages=n)

        L.append(('ok', f'<span style="color:#6b7280">{_ts()}</span>  '
                        f'✓ FAISS · {index.ntotal} vectors'))
        if all_tables:
            L.append(('ok', f'<span style="color:#6b7280">{_ts()}</span>  '
                            f'✓ {len(all_tables)} table(s) → CSV export ready'))
        L.append(('inf', f'<span style="color:#6b7280">{_ts()}</span>  '
                         f'<span style="color:#22c55e">⬤ Ready</span>  — '
                         f'Q&A unlocked for <b style="color:#e2e4eb">{_esc(fname)}</b>'))

        yield (gr.update(value=_log(L,100,'ready','ok'), visible=True),
               gr.update(value=zp, visible=zp is not None),
               gr.update(visible=True),
               gr.update(visible=False),
               [],   # reset chatbot
               '')   # reset sources

    except Exception as exc:
        traceback.print_exc()
        L.append(('err', f'<span style="color:#6b7280">{_ts()}</span>  ✗ {_esc(str(exc))}'))
        yield emit(L, 0, 'error', tone='err')


def on_transcribe(audio):
    """Called when microphone recording stops → puts text into question_in."""
    if not audio:
        return gr.update()
    text = transcribe(audio)
    if text:
        return gr.update(value=text)
    return gr.update()


def on_ask(question, history):
    """
    Chatbot generator.
    Outputs: (chatbot_history, sources_html, question_in_clear)

    FLOW:
      0. Chitchat detection  →  friendly reply, skip everything
      1. Add user message, show cycling thinking animation
      2. FAISS search (if document loaded)
      3. Intent detection  →  'document' | 'general' | 'web'
      4. Build prompt:
           document → raw text chunks + matched CSV tables (structured data)
           general  → LLM knowledge, friendly tone
           web      → DuckDuckGo snippets + LLM
      5. Stream tokens back into chat
      6. Render source panel (passages + inline tables)
    """
    if not question or not question.strip():
        yield history, '', gr.update(), gr.update()
        return

    # ── Step 0: chitchat guard ────────────────────────────────────────────────
    if _is_chitchat(question):
        history = list(history or []) + [[question, None]]
        history[-1][1] = "💬 Chatting…"
        yield history, '', gr.update(value=''), gr.update()
        chitchat_prompt = (
            'You are a friendly, helpful document assistant. '
            'The user is making small talk or greeting you. '
            'Respond naturally and warmly in 1-2 sentences. '
            'You can mention you are ready to help them query their document if one is loaded.\n\n'
            f'User: {question}'
        )
        response = ''
        for token in _stream_qa(chitchat_prompt):
            response += token
            history[-1][1] = response
            yield history, '', gr.update(), gr.update()
        badge_map = {'chitchat': ('💬', '#9ca3af', 'Casual chat')}
        sources_html = (
            _TABLE_CSS +
            '<div style="padding:6px 12px;border:1px solid #9ca3af40;border-radius:6px;'
            'background:#9ca3af10;font-family:monospace;font-size:12px;color:#9ca3af">'
            '💬 Casual chat — no document search performed</div>'
        )
        _save_turn(S.get('name', ''), question, response, 'chitchat')
        yield history, sources_html, gr.update(), _render_history_html()
        return

    history = list(history or []) + [[question, None]]

    # ── Step 1: show animated thinking ──────────────────────────────────────
    for msg in THINKING_MSGS[:3]:
        history[-1][1] = msg
        yield history, '', gr.update(value=''), gr.update()
        time.sleep(0.18)

    # ── Step 2: FAISS search ─────────────────────────────────────────────────
    hits_with_scores = []
    if S['index'] is not None:
        history[-1][1] = "🔍 Searching document vectors…"
        yield history, '', gr.update(), gr.update()
        hits_with_scores = search_with_scores(question, S['index'], S['chunks'])

    # ── Step 3: intent ───────────────────────────────────────────────────────
    intent = _detect_intent(hits_with_scores, question)

    # ── Step 4: build prompt + stream ────────────────────────────────────────
    web_snippets = []

    if intent == 'document':
        is_synthesis = _is_synthesis_query(question)
        history[-1][1] = "📄 Gathering all document data…" if is_synthesis else "📄 Answering from document…"
        yield history, '', gr.update(), gr.update()

        # Raw text context from FAISS hits
        text_context = '\n\n---\n\n'.join(c for c, _ in hits_with_scores) if hits_with_scores else ''

        # ── CROSS-PAGE TABLE INJECTION ────────────────────────────────────
        # For synthesis queries: inject ALL tables across all pages so the
        # LLM can do cross-page joins (e.g. animal data on page 1/3 +
        # diagnosis data on page 2/4).
        # For regular queries: only inject tables from pages in FAISS hits.
        table_context = ''
        if is_synthesis:
            # Inject ALL tables — synthesis needs the full picture
            for t in S['tables']:
                try:
                    df = pd.read_csv(t['path'])
                    table_context += (
                        f'\n[STRUCTURED TABLE — Page {t["page"]}]\n'
                        + df.to_string(index=False)
                        + '\n'
                    )
                except Exception:
                    pass
        else:
            # Regular query: inject tables from pages that scored in FAISS
            hit_pages = set()
            for chunk, _ in hits_with_scores:
                m = re.search(r'\[Page (\d+)\]', chunk)
                if m: hit_pages.add(int(m.group(1)))
            for t in S['tables']:
                if t['page'] in hit_pages:
                    try:
                        df = pd.read_csv(t['path'])
                        table_context += (
                            f'\n[STRUCTURED TABLE — Page {t["page"]}]\n'
                            + df.to_string(index=False)
                            + '\n'
                        )
                    except Exception:
                        pass

        full_context = text_context
        if table_context:
            full_context += '\n\n' + table_context

        # ── MARKDOWN TABLE INSTRUCTION ────────────────────────────────────
        table_format_instruction = (
            'IMPORTANT FORMATTING RULES:\n'
            '- When the answer contains tabular data (lists of people, animals, numbers, etc.), '
            'you MUST format it as a **Markdown table** using | column | header | syntax.\n'
            '- Include ALL rows from the data — do not truncate or summarize.\n'
            '- Use clear column headers that match the original data columns.\n'
            '- If the user asks to "make a table" or "list all", always output a markdown table.\n'
            '- You can combine data from multiple pages into one unified table.\n'
            '- If data from different pages has related columns (e.g. owner names, animal types, '
            'diagnoses), join/merge them into a single comprehensive table.\n\n'
        )

        prompt = (
            'You are a helpful assistant analysing a handwritten document. '
            'Answer the question using the document excerpts and structured table data below. '
            'Be concise and factual. Prefer the TABLE data over raw OCR text when both are present — '
            'the table is cleaner. '
            'If the answer is genuinely not in the data, say so naturally and concisely '
            '(do NOT prefix every answer with "Not found in document").\n\n'
            + table_format_instruction
            + f'Document data:\n{full_context}\n\n'
            f'Question: {question}'
        )

    elif intent == 'general':
        history[-1][1] = "🧠 Using general knowledge…"
        yield history, '', gr.update(), gr.update()
        doc_hint = f' (User has uploaded: "{S["name"]}")' if S['name'] else ''
        # If there are weak hits, still pass them — maybe the threshold was just barely missed
        weak_ctx = ''
        if hits_with_scores:
            weak_ctx = (
                '\n\nWeak document matches (may or may not be relevant):\n'
                + '\n---\n'.join(c[:300] for c, _ in hits_with_scores[:2])
            )
        # Also inject table data if available — general intent may still benefit
        table_hint = ''
        if S.get('tables'):
            for t in S['tables']:
                try:
                    df = pd.read_csv(t['path'])
                    table_hint += (
                        f'\n[TABLE — Page {t["page"]}]\n'
                        + df.to_string(index=False)
                        + '\n'
                    )
                except Exception:
                    pass
            if table_hint:
                weak_ctx += '\n\nExtracted tables from document:' + table_hint
        prompt = (
            f'You are a friendly and knowledgeable assistant.{doc_hint} '
            f'Answer helpfully from your general knowledge.{weak_ctx}\n\n'
            f'Question: {question}\n\n'
            'FORMATTING: If the answer contains tabular data, format it as a Markdown table.\n\n'
            'If the question seems like it could be about their document, add one sentence '
            'at the end suggesting they ask more specifically.'
        )

    else:  # 'web'
        history[-1][1] = "🌐 Searching the web…"
        yield history, '', gr.update(), gr.update()
        time.sleep(0.4)

        web_snippets = _ddg_search(question)
        web_ctx = '\n'.join(f'- {s}' for s in web_snippets) if web_snippets else ''
        prompt = (
            'You are a helpful assistant. '
            'Answer the question below using the web snippets if helpful, '
            'otherwise use your general knowledge. Be conversational and concise.\n\n'
            f'Web snippets:\n{web_ctx}\n\nQuestion: {question}'
            if web_ctx else
            f'You are a helpful assistant. Answer this question:\n\n{question}'
        )

    # ── Step 5: Stream tokens ─────────────────────────────────────────────────
    response = ''
    try:
        for token in _stream_qa(prompt):
            response += token
            history[-1][1] = response
            yield history, '', gr.update(), gr.update()
    except Exception as e:
        history[-1][1] = f'⚠️ Error: {e}'
        yield history, '', gr.update(), gr.update()
        return

    # ── Step 6: render source panel ──────────────────────────────────────────
    # For synthesis queries, show ALL tables in source panel regardless of FAISS hits
    display_tables = S['tables'] if (_is_synthesis_query(question) and S.get('tables')) else S['tables']
    sources_html = _format_sources_html(hits_with_scores, display_tables, intent, web_snippets)
    _save_turn(S.get('name', ''), question, response, intent)
    yield history, sources_html, gr.update(), _render_history_html()


# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 — CSS + Gradio UI
# ══════════════════════════════════════════════════════════════════════════════
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,700;0,9..144,900;1,9..144,300&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Kill Gradio chrome ── */
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
.gradio-container label,
.gradio-container .label-wrap > span,
.gradio-container p {{ color:var(--text) !important; font-family:'Inter',sans-serif !important; }}

.gradio-container textarea,
.gradio-container input[type=text] {{
    background:var(--sur2) !important; border:1px solid var(--bdr) !important;
    color:var(--text) !important; border-radius:8px !important;
    font-family:'Inter',sans-serif !important; font-size:14px !important;
}}
.gradio-container textarea:focus, .gradio-container input:focus {{
    border-color:var(--amber) !important;
    box-shadow:0 0 0 3px rgba(240,165,0,.1) !important; outline:none !important;
}}
.gradio-container button.primary {{
    background:var(--amber) !important; color:#0d0f14 !important;
    border:none !important; border-radius:8px !important;
    font-family:'Inter',sans-serif !important; font-weight:600 !important;
    font-size:14px !important; padding:10px 20px !important;
    transition:opacity .15s,transform .1s !important; box-shadow:none !important;
}}
.gradio-container button.primary:hover  {{ opacity:.88 !important; transform:translateY(-1px) !important; }}
.gradio-container button.primary:active {{ transform:translateY(0) !important; }}
.gradio-container button:not(.primary) {{
    background:var(--sur2) !important; border:1px solid var(--bdr) !important;
    color:var(--text) !important; border-radius:8px !important;
}}
.gradio-container .file-preview,
.gradio-container [data-testid="file"] {{
    background:var(--sur2) !important; border:1.5px dashed var(--bdr) !important;
    border-radius:var(--r) !important; color:var(--muted) !important; transition:.15s !important;
}}
.gradio-container [data-testid="file"]:hover,
.gradio-container .file-preview:hover {{
    border-color:var(--amber) !important; background:rgba(240,165,0,.03) !important;
}}
.gradio-container [data-testid="audio"], .gradio-container .audio-recorder {{
    background:var(--sur2) !important; border:1px solid var(--bdr) !important;
    border-radius:var(--r) !important;
}}

/* ═══════════════════════════════════════════════════
   CHAT BUBBLES — editorial, not generic AI slop
   ═══════════════════════════════════════════════════ */

/* Wipe Gradio's chatbot container chrome */
.gradio-container .chatbot {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}
.gradio-container .message-bubble-border,
.gradio-container .message .bubble-wrap {{
    box-shadow: none !important;
}}

/* ── USER bubble: right-aligned, amber left-rule, dark bg ── */
.gradio-container .message.user {{
    justify-content: flex-end !important;
    margin-bottom: 20px !important;
    padding: 0 !important;
}}
.gradio-container .message.user .message-bubble-border {{
    background: #17192400 !important;
    border: none !important;
    border-right: 3px solid var(--amber) !important;
    border-radius: 10px 2px 2px 10px !important;
    padding: 9px 14px 9px 12px !important;
    max-width: 68% !important;
    color: #c8cde0 !important;
    font-size: 13.5px !important;
    line-height: 1.65 !important;
    font-weight: 400 !important;
    background: #15171f !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ── BOT bubble: left-aligned, amber left-rule on hover, no box ── */
.gradio-container .message.bot {{
    justify-content: flex-start !important;
    margin-bottom: 20px !important;
    padding: 0 !important;
}}
.gradio-container .message.bot .message-bubble-border {{
    background: transparent !important;
    border: none !important;
    border-left: 2px solid #2a2e42 !important;
    border-radius: 0 !important;
    padding: 6px 14px !important;
    max-width: 96% !important;
    color: #c8cde0 !important;
    font-size: 13.5px !important;
    line-height: 1.8 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-left-color .2s ease !important;
}}
.gradio-container .message.bot:hover .message-bubble-border {{
    border-left-color: var(--amber) !important;
}}

/* Hide avatars — generic */
.gradio-container .message .avatar-container,
.gradio-container .avatar-container,
.gradio-container .icon-wrap {{ display:none !important; }}

/* Copy button: only on hover */
.gradio-container .message .copy-btn,
.gradio-container .message [title="Copy"] {{
    opacity:0 !important; transition:opacity .15s !important;
    background:transparent !important; border:none !important; color:var(--muted) !important;
}}
.gradio-container .message:hover .copy-btn,
.gradio-container .message:hover [title="Copy"] {{ opacity:1 !important; }}

/* Chatbot scroll area */
.gradio-container .chatbot .wrap,
.gradio-container .chatbot > div {{ background:transparent !important; }}

/* ═══════════════════════════════════════════════════
   MARKDOWN TABLES in bot bubbles — premium dark theme
   ═══════════════════════════════════════════════════ */
.gradio-container .message.bot table {{
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 12.5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    margin: 12px 0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #23273a !important;
}}
.gradio-container .message.bot thead th {{
    background: #1e2130 !important;
    color: #f0a500 !important;
    padding: 8px 12px !important;
    border: 1px solid #23273a !important;
    text-align: left !important;
    font-weight: 600 !important;
    font-size: 11.5px !important;
    letter-spacing: 0.02em !important;
    white-space: nowrap !important;
}}
.gradio-container .message.bot tbody td {{
    background: #13151c !important;
    color: #c8cde0 !important;
    padding: 6px 12px !important;
    border: 1px solid #1e2130 !important;
    white-space: nowrap !important;
    font-size: 12px !important;
}}
.gradio-container .message.bot tbody tr:hover td {{
    background: #191c25 !important;
}}
.gradio-container .message.bot .message-bubble-border {{
    overflow-x: auto !important;
}}

/* ─────────────────────────────────────────────────── */

/* Layout */
#root {{ max-width:1340px; margin:0 auto; padding:28px 22px 60px; }}

/* Hero */
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

/* Panels */
.panel {{ background:var(--sur); border:1px solid var(--bdr); border-radius:12px; padding:18px; overflow:hidden; }}
.plabel {{
    display:block; font-size:10px; font-weight:600; letter-spacing:.14em;
    text-transform:uppercase; font-family:'JetBrains Mono',monospace;
    color:var(--muted); padding-bottom:12px; margin-bottom:16px;
    border-bottom:1px solid var(--bdr);
}}

/* Stages */
.stages {{ display:flex; flex-direction:column; }}
.stage {{ display:flex; align-items:center; gap:11px; padding:7px 0; border-bottom:1px solid var(--sur2); }}
.stage:last-child {{ border-bottom:none; }}
.snum {{ width:20px;height:20px;border-radius:50%;background:var(--sur2);border:1px solid var(--bdr);
         display:flex;align-items:center;justify-content:center;font-size:9px;
         font-family:'JetBrains Mono',monospace;color:var(--muted);flex-shrink:0; }}
.sname {{ font-size:13px; font-weight:600; color:var(--text); }}
.sdesc {{ font-size:11.5px; color:var(--muted); margin-top:1px; }}

/* Idle */
.idle {{ display:flex;flex-direction:column;align-items:center;justify-content:center;
         min-height:340px;gap:12px;text-align:center;padding:40px 24px; }}
.idle-txt {{ font-size:13.5px;color:var(--muted);max-width:200px;line-height:1.65; }}

/* Source panel */
.src-panel {{
    max-height:500px; overflow-y:auto; padding:4px 0;
    scrollbar-width:thin; scrollbar-color:var(--bdr) transparent;
}}
.src-panel::-webkit-scrollbar {{ width:5px; }}
.src-panel::-webkit-scrollbar-thumb {{ background:var(--bdr); border-radius:3px; }}

/* History panel */
.hist-panel {{
    max-height:200px; overflow-y:auto; padding:2px 0;
    scrollbar-width:thin; scrollbar-color:var(--bdr) transparent;
}}
.hist-panel::-webkit-scrollbar {{ width:4px; }}
.hist-panel::-webkit-scrollbar-thumb {{ background:var(--bdr); border-radius:3px; }}
.hist-panel details summary::-webkit-details-marker {{ display:none; }}

/* Input row */
.input-row {{ display:flex; gap:8px; align-items:flex-end; }}
.divider {{ height:1px;background:var(--bdr);margin:14px 0; }}

@media(max-width:700px){{
    .hero{{flex-direction:column;align-items:flex-start;}}
    .hero-title{{font-size:26px;}}
    #root{{padding:16px 12px 40px;}}
}}
"""


# ── Build UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(primary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.slate),
    css=CSS, title='Document Q&A',
) as app:

    gr.HTML(f"""
    <div id="root">
      <div class="hero">
        <div>
          <div class="hero-eyebrow">Document Intelligence</div>
          <h1 class="hero-title">Read. Search. <em>Answer.</em></h1>
          <p class="hero-sub">Upload a scanned or handwritten document — every page is
             extracted, indexed, and made queryable by text or voice.<br>
             Ask anything: document, general knowledge, or web search.</p>
        </div>
        <div class="hero-right">
          <span class="pill"><span class="dot-g"></span>System ready</span>
          <span class="pill"><span class="dot-b"></span>{BACKEND_LABEL}</span>
          <span class="pill">FAISS · local vectors</span>
          <span class="pill">Max 100 pages</span>
        </div>
      </div>
    </div>""")

    with gr.Row(elem_id="root", equal_height=False):

        # ── LEFT: Upload + History panel ─────────────────────────────────────
        with gr.Column(scale=4, min_width=280, elem_classes="panel"):
            gr.HTML('<span class="plabel">01 · Upload Document</span>')
            file_in     = gr.File(label="Drop PDF or image  (max 100 pages)",
                                  file_types=[".pdf",".png",".jpg",".jpeg",".tiff"], height=120)
            process_btn = gr.Button("Analyse Document", variant="primary", size="lg")
            status_box  = gr.HTML(visible=False)
            dl_btn      = gr.File(label="Download extracted tables (CSV zip)",
                                  visible=False, interactive=False)
            gr.HTML("""
            <div class="divider"></div>
            <div class="stages">
              <div class="stage"><div class="snum">1</div>
                <div><div class="sname">Decode</div><div class="sdesc">Rasterise pages at 150 DPI</div></div></div>
              <div class="stage"><div class="snum">2</div>
                <div><div class="sname">Read</div><div class="sdesc">Extract text, tables, layout</div></div></div>
              <div class="stage"><div class="snum">3</div>
                <div><div class="sname">Embed</div><div class="sdesc">384-dim vectors (local CPU)</div></div></div>
              <div class="stage"><div class="snum">4</div>
                <div><div class="sname">Index</div><div class="sdesc">FAISS nearest-neighbour</div></div></div>
              <div class="stage"><div class="snum">5</div>
                <div><div class="sname">Ready</div><div class="sdesc">Q&amp;A unlocked</div></div></div>
            </div>
            <div class="divider"></div>""")

            # ── Chat History ─────────────────────────────────────────────────
            gr.HTML(
                '<span style="display:block;font-size:10px;font-weight:600;letter-spacing:.14em;'
                'text-transform:uppercase;font-family:\'JetBrains Mono\',monospace;color:#374151;'
                'margin-bottom:10px">Past Sessions</span>'
            )
            history_panel = gr.HTML(
                value=_render_history_html(),
                elem_classes="hist-panel",
            )

        # ── RIGHT: Chat panel ─────────────────────────────────────────────────
        with gr.Column(scale=7, elem_classes="panel"):
            gr.HTML('<span class="plabel">02 · Chat with Document</span>')

            idle_html = gr.HTML("""
            <div class="idle">
              <div style="opacity:.12">
                <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                  <rect x="1.5" y="1.5" width="49" height="49" rx="10" stroke="#dde1ec" stroke-width="1.2"/>
                  <path d="M15 19h22M15 26h15M15 33h10" stroke="#dde1ec" stroke-width="1.2" stroke-linecap="round"/>
                </svg>
              </div>
              <div class="idle-txt">Process a document on the left to unlock chat.<br><br>
              Greetings, document questions, and general knowledge all work.</div>
            </div>""", visible=True)

            qa_col = gr.Column(visible=False)
            with qa_col:
                chatbot = gr.Chatbot(
                    label="",
                    height=480,
                    show_copy_button=True,
                    bubble_full_width=False,
                    show_label=False,
                    render_markdown=True,
                    sanitize_html=False,
                )

                with gr.Row(elem_classes="input-row"):
                    question_in = gr.Textbox(
                        label="",
                        placeholder="Ask anything… Enter to send · or record below",
                        lines=1,
                        scale=8,
                        container=False,
                    )
                    ask_btn = gr.Button("↑", variant="primary", scale=1, min_width=48)

                audio_in = gr.Audio(
                    label="🎤  Voice — transcription appears in the box above",
                    type="filepath",
                    sources=["microphone"],
                    max_length=30,
                )

                gr.HTML('<div class="divider"></div>')
                gr.HTML(
                    '<span style="font-size:10px;font-family:monospace;color:#374151;'
                    'letter-spacing:.1em;text-transform:uppercase">Sources &amp; retrieved tables</span>'
                )
                sources_out = gr.HTML('', elem_classes="src-panel")

    # ── Event wiring ──────────────────────────────────────────────────────────

    process_btn.click(
        fn=on_upload,
        inputs=[file_in],
        outputs=[status_box, dl_btn, qa_col, idle_html, chatbot, sources_out],
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
        inputs=[question_in, chatbot],
        outputs=[chatbot, sources_out, question_in, history_panel],
        queue=True,
        show_progress="hidden",
    )
    ask_btn.click(**_ask_cfg)
    question_in.submit(**_ask_cfg)


    print('Launching…')
    app.queue(default_concurrency_limit=2)
    app.launch(server_name='0.0.0.0', server_port=7860,
               share=GRADIO_SHARE, inbrowser=OPEN_BROWSER)

if __name__ == '__main__':
    print('Launching…')
    app.queue(default_concurrency_limit=2)
    app.launch(server_name='0.0.0.0', server_port=7860,
               share=GRADIO_SHARE, inbrowser=OPEN_BROWSER)
