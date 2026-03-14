"""
indexer.py — Build a LlamaIndex VectorStoreIndex from OCR output.

This is where our custom OCR pipeline hands off to LlamaIndex.

Flow:
  OCR pages (list of dicts) + table row chunks
    → LlamaIndex TextNode objects (with page/doc metadata)
    → SentenceSplitter (token-aware, sentence-boundary-preserving)
    → VectorStoreIndex (in-memory, using BGE-small embeddings)

The metadata on every node — page_label, doc_name, is_table_row — is what
powers the source panel: we know exactly which page and document each
retrieved chunk came from.
"""

import re
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL

# ── Global embedding model (set once, used by LlamaIndex everywhere) ──────────
print(f"Loading embedding model: {EMBED_MODEL}")
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
print("Embedding model ready (384-dim, local CPU)")

# SentenceSplitter: token-based, respects sentence boundaries, handles
# research papers (dense prose) and registers (short rows) equally well.
_splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    paragraph_separator="\n\n",   # honour double-newline paragraph breaks
)


def build_index(
    pages: list,
    tables: list,
    doc_name: str,
) -> VectorStoreIndex:
    """
    Build and return a VectorStoreIndex for one document.

    Args:
        pages:     list of {'text': str, 'page': int} from ocr_page()
        tables:    list of table metadata dicts (with 'page', 'path')
        doc_name:  original filename used for metadata tagging

    Returns:
        VectorStoreIndex (in-memory, immediately queryable)
    """
    nodes = []

    # ── Text chunks from OCR pages ───────────────────────────────────────────
    for page_data in pages:
        page_text = page_data["text"].strip()
        visual_summary = (page_data.get("visual_summary") or "").strip()
        if not page_text and not visual_summary:
            continue

        if page_text:
            # Wrap as a LlamaIndex Document with metadata that propagates to chunks
            doc = Document(
                text=page_text,
                metadata={
                    "page_label": str(page_data["page"]),
                    "doc_name":   doc_name,
                    "is_table_row": False,
                    "is_visual_summary": False,
                },
                # Exclude page_label / doc_name from the embedding so they don't
                # bias the vector, but keep them in the text sent to the LLM.
                excluded_embed_metadata_keys=["is_table_row", "is_visual_summary"],
            )
            page_nodes = _splitter.get_nodes_from_documents([doc])
            nodes.extend(page_nodes)

        if visual_summary:
            nodes.append(
                TextNode(
                    text=f"[Doc: {doc_name}][Page {page_data['page']}][VISUAL SUMMARY] {visual_summary}",
                    metadata={
                        "page_label": str(page_data["page"]),
                        "doc_name": doc_name,
                        "is_table_row": False,
                        "is_visual_summary": True,
                    },
                )
            )

    # ── Table row nodes ───────────────────────────────────────────────────────
    # Each row gets its own node so FAISS can find individual cell values.
    # Example node text:
    #   "[Doc: register.pdf][Page 1][TABLE ROW] Owner: M. Abid | Animal: Calf"
    import pandas as pd

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
                    row_text = f"[Doc: {doc_name}][Page {t['page']}][TABLE ROW] {parts}"
                    node = TextNode(
                        text=row_text,
                        metadata={
                            "page_label":   str(t["page"]),
                            "doc_name":     doc_name,
                            "is_table_row": True,
                            "is_visual_summary": False,
                        },
                    )
                    nodes.append(node)
        except Exception as e:
            print(f"  Table row chunk error (page {t.get('page')}): {e}")

    if not nodes:
        raise ValueError(
            "No indexable content produced — check OCR output and ensure "
            "the document has readable text."
        )

    print(f"Building index: {len(nodes)} nodes ({doc_name})")
    index = VectorStoreIndex(nodes, show_progress=True)
    print(f"Index ready: {len(nodes)} nodes")
    return index
