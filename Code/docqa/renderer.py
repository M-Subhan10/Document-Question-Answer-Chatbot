"""
renderer.py — HTML generation for the Gradio UI.

Produces:
  render_log(lines, pct, stage, tone)           — macOS-style terminal log
  render_sources(source_nodes, tables)          — retrieved passages + tables
  TABLE_CSS                                     — shared table styles
"""

import re
import pandas as pd
from html import escape as _esc
from datetime import datetime


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL LOG
# ══════════════════════════════════════════════════════════════════════════════

def render_log(lines: list, pct: int, stage: str, tone: str = "run") -> str:
    """Render a macOS-style terminal panel with animated progress bar."""
    bar  = {"run": "#f0a500", "ok": "#22c55e", "err": "#ef4444"}.get(tone, "#f0a500")
    anim = "animation:sh 1.4s linear infinite;background-size:300% 100%;" if tone == "run" else ""

    rows = ""
    for kind, txt in lines[-18:]:
        c = {"ok": "#22c55e", "run": "#f0a500", "err": "#ef4444",
             "dim": "#374151", "inf": "#60a5fa"}.get(kind, "#9ca3af")
        rows += f'<div style="padding:1px 0;color:{c};line-height:1.75">{txt}</div>'

    return (
        f'<div style="border:1px solid #1e2130;border-radius:8px;overflow:hidden;'
        f'background:#0c0e14;font-family:\'JetBrains Mono\',monospace;font-size:12px">'
        f'<div style="display:flex;align-items:center;gap:6px;padding:7px 12px;'
        f'background:#131520;border-bottom:1px solid #1e2130">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:#ff5f57;display:inline-block"></span>'
        f'<span style="width:9px;height:9px;border-radius:50%;background:#febc2e;display:inline-block"></span>'
        f'<span style="width:9px;height:9px;border-radius:50%;background:#28c840;display:inline-block"></span>'
        f'<span style="margin-left:8px;font-size:10px;letter-spacing:.1em;text-transform:uppercase;'
        f'color:#374151">{_esc(stage)}</span></div>'
        f'<div style="padding:10px 14px;min-height:64px;max-height:240px;overflow-y:auto">{rows}</div>'
        f'<div style="padding:7px 14px 11px;background:#131520;border-top:1px solid #1e2130">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;'
        f'font-size:10px;letter-spacing:.1em;text-transform:uppercase">'
        f'<span style="color:#374151">Progress</span>'
        f'<span style="color:{bar}">{pct}%</span></div>'
        f'<div style="height:3px;background:#1e2130;border-radius:3px;overflow:hidden">'
        f'<div style="height:100%;width:{pct}%;background:{bar};border-radius:3px;'
        f'transition:width .3s ease;{anim}"></div></div></div></div>'
        f'<style>@keyframes sh{{0%{{background-position:100% 0}}100%{{background-position:-100% 0}}}}</style>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# TABLE HTML
# ══════════════════════════════════════════════════════════════════════════════

TABLE_CSS = """
<style>
.dtable{border-collapse:collapse;width:100%;font-size:12px;
        font-family:'JetBrains Mono',monospace;margin:0}
.dtable th{background:#1e2130;color:#f0a500;padding:5px 10px;
           border:1px solid #23273a;text-align:left;font-weight:600;white-space:nowrap}
.dtable td{background:#13151c;color:#dde1ec;padding:4px 10px;
           border:1px solid #1e2130;white-space:nowrap}
.dtable tr:hover td{background:#191c25}
</style>"""


def _df_to_html(df: pd.DataFrame) -> str:
    ths  = "".join(f"<th>{_esc(str(c))}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        tds  = "".join(f"<td>{_esc(str(v))}</td>" for v in row)
        body += f"<tr>{tds}</tr>"
    return f'<table class="dtable"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_sources(source_nodes: list, all_tables: list) -> str:
    """
    Render retrieved passages and their matching page tables as styled HTML.

    source_nodes: list of NodeWithScore from LlamaIndex (post stream_chat)
    all_tables:   list of table metadata dicts from ocr.py
    """
    parts = [TABLE_CSS]

    if not source_nodes:
        parts.append(
            '<div style="color:#5a6278;font-size:12px;font-family:monospace;padding:8px">'
            "No document passages retrieved for this query.</div>"
        )
        return "".join(parts)

    # ── Intent badge (document = green, since sources panel only shows on doc queries)
    parts.append(
        '<div style="margin-bottom:14px;padding:6px 12px;'
        'border:1px solid #22c55e40;border-radius:6px;background:#22c55e10;'
        'font-family:monospace;font-size:12px;color:#22c55e">'
        "📄 Answered from document</div>"
    )

    shown_tables: set = set()

    for i, node_with_score in enumerate(source_nodes, 1):
        node  = node_with_score.node
        score = node_with_score.score or 0.0

        page  = node.metadata.get("page_label", "?")
        doc   = node.metadata.get("doc_name", "")
        is_tr = node.metadata.get("is_table_row", False)
        is_visual = node.metadata.get("is_visual_summary", False)

        # Clean display text: strip our internal tags
        clean = re.sub(r"\[(?:Doc:[^\]]+|Page \d+|TABLE ROW|HEADER|SECTION|FIGURE|NOTES|TABLE)\]", "", node.text)
        clean = clean.strip()

        pct   = int(min(score, 1.0) * 100)

        # Table row nodes: show a tighter pill style
        if is_tr:
            parts.append(f"""
            <div style="margin-bottom:8px;border:1px solid #f0a50030;border-radius:6px;
                        padding:6px 10px;background:#0f1118">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">
                <span style="font-family:monospace;font-size:10px;color:#f0a500">
                  📊 Table Row · Page {_esc(str(page))}
                </span>
                <span style="font-family:monospace;font-size:10px;color:#374151">{pct}%</span>
              </div>
              <div style="font-size:11.5px;color:#9ca3af;line-height:1.6;
                          font-family:monospace">{_esc(clean[:300])}</div>
            </div>""")
        elif is_visual:
            parts.append(f"""
            <div style="margin-bottom:10px;border:1px solid #60a5fa30;border-radius:6px;
                        padding:8px 10px;background:#0f1118">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                <span style="font-family:monospace;font-size:10px;color:#60a5fa">
                  👁 Visual Summary · Page {_esc(str(page))}
                </span>
                <span style="font-family:monospace;font-size:10px;color:#374151">{pct}%</span>
              </div>
              <div style="font-size:12px;color:#cbd5e1;line-height:1.7">{_esc(clean[:420])}</div>
            </div>""")
        else:
            parts.append(f"""
            <div style="margin-bottom:14px;border:1px solid #23273a;border-radius:8px;overflow:hidden">
              <div style="background:#131520;padding:5px 12px;display:flex;align-items:center;gap:10px">
                <span style="font-family:monospace;font-size:11px;color:#5a6278">
                  📄 Passage {i} · Page {_esc(str(page))}
                </span>
                <span style="flex:1;height:3px;background:#1e2130;border-radius:3px;overflow:hidden">
                  <span style="display:block;height:100%;width:{pct}%;background:#f0a500;border-radius:3px"></span>
                </span>
                <span style="font-family:monospace;font-size:10px;color:#f0a500">{pct}%</span>
              </div>
              <div style="padding:8px 12px;font-size:12px;line-height:1.75;color:#9ca3af;
                          background:#13151c;white-space:pre-wrap;max-height:110px;overflow-y:auto"
                   >{_esc(clean[:500])}{"…" if len(clean) > 500 else ""}</div>
            </div>""")

        # Render CSV tables for this page (if not already shown)
        try:
            pg_int = int(page)
        except ValueError:
            continue

        for t in all_tables:
            if t["page"] == pg_int and t["path"] not in shown_tables:
                shown_tables.add(t["path"])
                try:
                    df = pd.read_csv(t["path"])
                    parts.append(f"""
                    <div style="margin-bottom:14px;border:1px solid #f0a50060;
                                border-radius:8px;overflow:hidden">
                      <div style="background:#131520;padding:5px 12px;
                                  font-family:monospace;font-size:11px;color:#f0a500">
                        📊 Table · Page {pg_int} · {len(df)} rows × {len(df.columns)} cols
                      </div>
                      <div style="padding:8px;overflow-x:auto;background:#13151c">
                        {_df_to_html(df)}
                      </div>
                    </div>""")
                except Exception:
                    pass

    return "".join(parts)


def render_general_badge() -> str:
    return (
        TABLE_CSS
        + '<div style="padding:6px 12px;border:1px solid #60a5fa40;border-radius:6px;'
        'background:#60a5fa10;font-family:monospace;font-size:12px;color:#60a5fa">'
        "🧠 General knowledge — no document retrieval performed</div>"
    )


def render_chitchat_badge() -> str:
    return (
        TABLE_CSS
        + '<div style="padding:6px 12px;border:1px solid #9ca3af40;border-radius:6px;'
        'background:#9ca3af10;font-family:monospace;font-size:12px;color:#9ca3af">'
        "💬 Casual chat — no search performed</div>"
    )


def render_context_cleared_badge() -> str:
    return (
        TABLE_CSS
        + '<div style="padding:8px 12px;border:1px solid #ef444460;border-radius:6px;'
        'background:#ef444410;font-family:monospace;font-size:12px;color:#ef4444">'
        "⚠️ Context cleared — conversation history reset after 20 messages</div>"
    )


def render_warn_badge(remaining: int) -> str:
    return (
        TABLE_CSS
        + f'<div style="padding:8px 12px;border:1px solid #f0a50060;border-radius:6px;'
        f'background:#f0a50010;font-family:monospace;font-size:12px;color:#f0a500">'
        f"⚠️ {remaining} message{'s' if remaining != 1 else ''} until context reset (limit: 20)</div>"
    )


def render_overview_sources(page_nums: list[int], truncated: bool = False) -> str:
    pages = ", ".join(str(p) for p in page_nums) if page_nums else "document excerpts"
    note = " · truncated to fit context" if truncated else ""
    return (
        TABLE_CSS
        + '<div style="margin-bottom:14px;padding:6px 12px;'
        'border:1px solid #22c55e40;border-radius:6px;background:#22c55e10;'
        'font-family:monospace;font-size:12px;color:#22c55e">'
        f"📄 Overview answer built from pages: {pages}{note}</div>"
    )


def render_visual_sources(page_nums: list[int], truncated: bool = False) -> str:
    pages = ", ".join(str(p) for p in page_nums) if page_nums else "visual page notes"
    note = " · truncated to fit context" if truncated else ""
    return (
        TABLE_CSS
        + '<div style="margin-bottom:14px;padding:6px 12px;'
        'border:1px solid #60a5fa40;border-radius:6px;background:#60a5fa10;'
        'font-family:monospace;font-size:12px;color:#60a5fa">'
        f"👁 Visual answer built from pages: {pages}{note}</div>"
    )


def render_full_table_sources(all_tables: list, page_nums: list[int], truncated: bool = False) -> str:
    parts = [TABLE_CSS]
    pages = ", ".join(str(p) for p in page_nums) if page_nums else "all extracted tables"
    note = " · truncated to fit context" if truncated else ""
    parts.append(
        '<div style="margin-bottom:14px;padding:6px 12px;'
        'border:1px solid #22c55e40;border-radius:6px;background:#22c55e10;'
        'font-family:monospace;font-size:12px;color:#22c55e">'
        f"📊 Answered from full extracted tables on pages: {pages}{note}</div>"
    )

    for t in all_tables:
        if page_nums and t["page"] not in page_nums:
            continue
        try:
            df = pd.read_csv(t["path"])
            parts.append(f"""
            <div style="margin-bottom:14px;border:1px solid #f0a50060;
                        border-radius:8px;overflow:hidden">
              <div style="background:#131520;padding:5px 12px;
                          font-family:monospace;font-size:11px;color:#f0a500">
                📊 Table · Page {t["page"]} · {len(df)} rows × {len(df.columns)} cols
              </div>
              <div style="padding:8px;overflow-x:auto;background:#13151c">
                {_df_to_html(df)}
              </div>
            </div>""")
        except Exception:
            continue

    return "".join(parts)
