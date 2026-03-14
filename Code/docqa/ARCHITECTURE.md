# Architecture

## End-to-End Flow

1. Upload
- Accept up to 3 files per batch
- Support PDF and common image formats
- Enforce file-size and total-page caps early
- Process files independently so one bad file does not fail the whole batch

2. Document Routing
- PDF page with strong embedded text: use direct extraction
- PDF page with weak or missing embedded text: render that page and OCR it
- Standalone image: OCR it directly
- OCR can use a backup provider when the first pass looks weak

3. Normalization
- Clean null bytes and broken spacing
- Preserve page number, source type, and OCR confidence
- Parse pipe-style table output into CSV artifacts when present

4. Retrieval Index Build
- Document summary chunk
- One page-summary chunk per page
- Paragraph-level page-text chunks
- Table-row chunks
- Dense embeddings are built locally with `BAAI/bge-small-en-v1.5`

5. Query Planning
- Detect explicit page references like `page 2` or `pages 2 and 4`
- Detect broad overview questions like `what is this document about`
- Detect explicit cross-document references like `first document`, `second pdf`, `both files`
- Use active-document memory for follow-up questions like `what about page 3` or `and his phone number`

6. Retrieval
- Search all loaded documents unless the question narrows scope
- Score candidates with a hybrid signal:
  - dense similarity
  - lexical overlap
  - document/page memory bonuses
  - overview bonuses for document summaries
- Expand to neighboring page summaries around the strongest hits

7. Answering
- If documents are loaded, answer from document evidence by default
- If the user explicitly asks for an outside-the-doc answer, switch to general mode
- Prompt includes:
  - rolling conversation summary
  - recent verbatim turns
  - retrieved document context

8. Memory Management
- Keep the most recent turns verbatim
- Roll older turns into a compact memory summary
- Warn after 16 user turns
- Refresh the visible chat after 20 user turns
- Keep uploaded documents, retrieval state, and memory summary after refresh

## Why This Is More Robust

- Clean research papers no longer go through OCR unless the embedded text is weak
- Scanned pages and images still get OCR with a fallback path
- Broad questions do not depend on brittle intent detection
- Multi-page and follow-up questions reuse active page and document state
- One failed upload does not kill the whole session

## Expected Latency

Approximate per stage on a normal laptop / small server:

- App startup:
  - embedding model load: 3 to 8 seconds after the first install
- Upload processing:
  - clean digital PDF page: near-instant per page for extraction
  - OCR page: 2 to 8 seconds per page depending on provider and page complexity
  - 10-page clean paper: usually 5 to 15 seconds total
  - 10-page scanned form: usually 25 to 90 seconds total
- Question answering:
  - retrieval: under 1 second for up to 3 moderate-size documents
  - answer generation: 1 to 6 seconds depending on model

## Latency Tradeoffs

- More OCR fallback improves robustness but increases upload time
- Higher DPI improves OCR on handwriting and tiny text but increases OCR cost and latency
- More retrieval chunks improve recall but slightly increase embedding time during upload
- Conversation rollups improve long-chat precision while preventing prompt growth from slowing answers

## Known Limits

- Highly visual documents with charts and diagram-heavy answers still depend mostly on extracted text
- Very poor handwriting can still require manual confirmation
- Complex scientific layouts may extract well textually but lose some figure-level reasoning
- The current implementation keeps embeddings in memory and is sized for small-session use, not multi-tenant scale
