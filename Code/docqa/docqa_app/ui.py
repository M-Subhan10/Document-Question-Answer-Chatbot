from __future__ import annotations

from html import escape as esc

import gradio as gr

from .config import Settings
from .pipeline import DocumentPipeline
from .state import load_history


CSS = """
:root {
  --bg:#0d1117; --panel:#111827; --panel2:#161f2d; --text:#dbe3ee;
  --muted:#7f8ea3; --border:#273244; --accent:#f59e0b; --green:#22c55e; --red:#ef4444;
}
body, .gradio-container { background:var(--bg) !important; color:var(--text) !important; }
.gradio-container { max-width:1400px !important; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:18px; }
.section-label { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-bottom:10px; font-family:monospace; }
.notice { border:1px solid rgba(245,158,11,.35); background:rgba(245,158,11,.08); border-radius:10px; padding:10px 12px; font-size:13px; color:#fcd34d; }
.doc-card { border:1px solid var(--border); background:var(--panel2); border-radius:12px; padding:12px; margin-bottom:10px; }
.doc-title { font-weight:600; color:var(--text); margin-bottom:6px; }
.doc-meta { color:var(--muted); font-size:12px; font-family:monospace; margin-bottom:8px; }
.doc-summary { color:#c5d0de; font-size:13px; line-height:1.7; white-space:pre-wrap; }
.src-card { border:1px solid var(--border); background:var(--panel2); border-radius:12px; padding:12px; margin-bottom:10px; }
.src-head { color:#fbbf24; font-family:monospace; font-size:12px; margin-bottom:6px; }
.src-body { white-space:pre-wrap; color:#c5d0de; font-size:13px; line-height:1.7; max-height:180px; overflow:auto; }
.hist-card { border:1px solid var(--border); background:var(--panel2); border-radius:10px; padding:10px; margin-bottom:8px; }
.hist-head { color:var(--muted); font-family:monospace; font-size:11px; margin-bottom:6px; }
.hist-line { color:#c5d0de; font-size:12px; line-height:1.6; }
"""


def render_docs_panel(pipeline: DocumentPipeline) -> str:
    if not pipeline.state.docs:
        return '<div class="doc-summary">No documents loaded yet.</div>'
    parts = []
    for doc in pipeline.state.docs:
        warning_html = ""
        if doc.warnings:
            warning_html = "<br>".join(f"- {esc(item)}" for item in doc.warnings)
            warning_html = f'<div class="doc-meta" style="color:#fca5a5">{warning_html}</div>'
        parts.append(
            f"""
            <div class="doc-card">
              <div class="doc-title">{esc(doc.name)}</div>
              <div class="doc-meta">{doc.kind.upper()} · {doc.page_count} page(s)</div>
              {warning_html}
              <div class="doc-summary">{esc(doc.summary)}</div>
            </div>
            """
        )
    return "".join(parts)


def render_sources(hits) -> str:
    if not hits:
        return '<div class="doc-summary">No document evidence retrieved for this turn.</div>'
    parts = []
    for hit in hits[:12]:
        parts.append(
            f"""
            <div class="src-card">
              <div class="src-head">{esc(hit.chunk.doc_name)} · page {hit.chunk.page_num or 0} · {esc(hit.chunk.kind)} · {hit.score:.2f}</div>
              <div class="src-body">{esc(hit.chunk.text)}</div>
            </div>
            """
        )
    return "".join(parts)


def render_history(settings: Settings) -> str:
    sessions = list(reversed(load_history(settings)))
    if not sessions:
        return '<div class="doc-summary">No archived sessions yet.</div>'
    parts = []
    for session in sessions[:10]:
        docs = " + ".join(session.get("docs") or ["(no document)"])
        lines = []
        for turn in session.get("turns", [])[-3:]:
            lines.append(
                f'<div class="hist-line"><b>{esc(turn["time"])}</b> · Q: {esc(turn["q"][:90])}</div>'
            )
        parts.append(
            f"""
            <div class="hist-card">
              <div class="hist-head">{esc(session.get("date", ""))} · {esc(docs)}</div>
              {''.join(lines)}
            </div>
            """
        )
    return "".join(parts)


def build_app(settings: Settings | None = None):
    settings = settings or Settings()
    pipeline = DocumentPipeline(settings)

    def on_upload(files):
        if not files:
            return (
                '<div class="notice">Select between 1 and 3 PDF/image files.</div>',
                gr.update(visible=False),
                render_docs_panel(pipeline),
                render_history(settings),
                [],
                "",
                "",
            )
        try:
            docs, logs = pipeline.ingest(files)
        except Exception as exc:
            status = f"<div class='notice'>Upload failed: {esc(str(exc))}</div>"
            return (
                status,
                gr.update(value=None, visible=False),
                render_docs_panel(pipeline),
                render_history(settings),
                [],
                "",
                gr.update(value="", visible=False),
            )
        status = "<div class='notice'>" + "<br>".join(esc(line) for line in logs) + "</div>"
        return (
            status,
            gr.update(value=pipeline.state.zip_path, visible=bool(pipeline.state.zip_path)),
            render_docs_panel(pipeline),
            render_history(settings),
            [],
            "",
            gr.update(value="", visible=False),
        )

    def on_ask(question, history):
        if not question or not question.strip():
            yield history, "", gr.update(), render_history(settings), gr.update()
            return

        history = list(history or [])
        history.append([question, "Thinking…"])
        yield history, "", gr.update(value=""), render_history(settings), gr.update(value=pipeline.state.memory.notice or "", visible=bool(pipeline.state.memory.notice))

        latest_hits = []
        reset_chat = False
        try:
            for answer, hits, reset in pipeline.stream_answer(question):
                latest_hits = hits
                reset_chat = reset
                history[-1][1] = answer
                yield history, render_sources(hits), gr.update(value=""), render_history(settings), gr.update(value=pipeline.state.memory.notice or "", visible=bool(pipeline.state.memory.notice))
        except Exception as exc:
            history[-1][1] = f"Request failed: {exc}"
            yield history, render_sources(latest_hits), gr.update(value=""), render_history(settings), gr.update(value="The last request failed. You can retry or upload a cleaner file.", visible=True)
            return

        if reset_chat:
            archived_notice = pipeline.state.memory.notice or "Chat context was refreshed."
            history = [["System", archived_notice]]
            yield history, render_sources(latest_hits), gr.update(value=""), render_history(settings), gr.update(value=archived_notice, visible=True)

    with gr.Blocks(css=CSS, title="Document QA") as app:
        gr.HTML("<h1 style='margin:0 0 16px 0'>Document QA Pipeline</h1>")
        with gr.Row():
            with gr.Column(scale=4, elem_classes="panel"):
                gr.HTML('<div class="section-label">Upload</div>')
                file_in = gr.File(
                    label="Upload up to 3 PDFs or images",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"],
                    file_count="multiple",
                    height=130,
                )
                upload_btn = gr.Button("Process Documents", variant="primary")
                status_box = gr.HTML("")
                zip_out = gr.File(label="Extracted tables", visible=False, interactive=False)
                gr.HTML('<div class="section-label" style="margin-top:16px">Loaded documents</div>')
                docs_panel = gr.HTML(render_docs_panel(pipeline))
                gr.HTML('<div class="section-label" style="margin-top:16px">Archived sessions</div>')
                history_panel = gr.HTML(render_history(settings))

            with gr.Column(scale=8, elem_classes="panel"):
                gr.HTML('<div class="section-label">Chat</div>')
                notice_box = gr.HTML("", visible=False)
                chatbot = gr.Chatbot(height=520, bubble_full_width=False, show_copy_button=True)
                question_in = gr.Textbox(
                    label="",
                    placeholder="Ask about the uploaded documents. Broad questions like 'what is this document about?' are grounded automatically.",
                    lines=2,
                )
                ask_btn = gr.Button("Ask", variant="primary")
                gr.HTML('<div class="section-label" style="margin-top:14px">Sources</div>')
                sources_out = gr.HTML("")

        upload_btn.click(
            fn=on_upload,
            inputs=[file_in],
            outputs=[status_box, zip_out, docs_panel, history_panel, chatbot, sources_out, notice_box],
            queue=True,
            show_progress="hidden",
        )
        ask_btn.click(
            fn=on_ask,
            inputs=[question_in, chatbot],
            outputs=[chatbot, sources_out, question_in, history_panel, notice_box],
            queue=True,
            show_progress="hidden",
        )
        question_in.submit(
            fn=on_ask,
            inputs=[question_in, chatbot],
            outputs=[chatbot, sources_out, question_in, history_panel, notice_box],
            queue=True,
            show_progress="hidden",
        )

    return app
