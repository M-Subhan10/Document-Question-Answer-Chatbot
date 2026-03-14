"""
history.py — Disk-persistent session history for the sidebar panel.

The LlamaIndex ChatSummaryMemoryBuffer handles in-session conversation memory.
This module handles what the user sees in the "Past Sessions" sidebar:
  - Saves every completed Q&A turn to a JSON file on disk
  - Renders past sessions as collapsible HTML cards

The disk file survives server restarts if WORK_DIR is a persistent directory.
Set DOCQA_WORK_DIR to a permanent path (e.g. ~/docqa_data) to persist across restarts.
"""

import os, json, datetime
from html import escape as _esc
from config import WORK_DIR

HISTORY_FILE = os.path.join(WORK_DIR, "chat_history.json")


def _load() -> list:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_turn(doc_name: str, question: str, answer: str, intent: str = "document"):
    """Append one Q&A turn to the history file under today's session key."""
    sessions = _load()
    today    = datetime.date.today().isoformat()
    key      = f"{doc_name}::{today}"

    session  = next((s for s in sessions if s.get("key") == key), None)
    if session is None:
        session = {"key": key, "doc": doc_name or "(no document)", "date": today, "turns": []}
        sessions.append(session)

    session["turns"].append({
        "q":      question,
        "a":      answer,
        "intent": intent,
        "time":   datetime.datetime.now().strftime("%H:%M"),
    })

    # Cap disk file at 40 sessions
    if len(sessions) > 40:
        sessions = sessions[-40:]

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"History write error: {e}")


def render_history_html() -> str:
    """Render past sessions as read-only collapsible dark-theme HTML cards."""
    sessions = list(reversed(_load()))
    if not sessions:
        return (
            '<div style="padding:12px 4px;color:#374151;font-family:monospace;'
            'font-size:11px;text-align:center;line-height:1.9">'
            "No history yet.<br>Conversations appear here after each session.</div>"
        )

    _ico = {"document": "📄", "general": "🧠", "web": "🌐", "chitchat": "💬"}
    parts = []

    for si, sess in enumerate(sessions[:15]):
        turns = sess.get("turns", [])
        doc   = _esc(sess.get("doc", "?"))
        date  = sess.get("date", "")
        n     = len(turns)

        turns_html = ""
        for t in turns:
            ico  = _ico.get(t.get("intent", ""), "·")
            q    = _esc(str(t.get("q", ""))[:110])
            a    = _esc(str(t.get("a", ""))[:180])
            tm   = _esc(str(t.get("time", "")))
            qe   = q + ("…" if len(str(t.get("q", ""))) > 110 else "")
            ae   = a + ("…" if len(str(t.get("a", ""))) > 180 else "")
            turns_html += f"""
            <div style="padding:7px 0;border-bottom:1px solid #1a1d29">
              <div style="display:flex;gap:6px;align-items:baseline;margin-bottom:3px">
                <span style="font-size:9px;color:#374151;font-family:monospace">{tm}</span>
                <span style="font-size:9px">{ico}</span>
                <span style="font-size:12px;color:#dde1ec;font-weight:500">{qe}</span>
              </div>
              <div style="font-size:11px;color:#5a6278;line-height:1.6;padding-left:2px">{ae}</div>
            </div>"""

        is_open = "open" if si == 0 else ""
        parts.append(f"""
        <details style="margin-bottom:8px" {is_open}>
          <summary style="cursor:pointer;list-style:none;padding:8px 10px;
                          background:#131520;border:1px solid #23273a;border-radius:7px;
                          display:flex;align-items:center;gap:8px;user-select:none">
            <span style="font-size:10px;font-family:monospace;color:#f0a500">▶</span>
            <span style="flex:1;font-size:11px;color:#dde1ec;font-family:monospace;
                         white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{doc}</span>
            <span style="font-size:10px;color:#374151;font-family:monospace;white-space:nowrap"
                  >{date} · {n}Q</span>
          </summary>
          <div style="border:1px solid #1e2130;border-top:none;border-radius:0 0 7px 7px;
                      padding:4px 10px;background:#0c0e14">
            {turns_html}
          </div>
        </details>""")

    return "".join(parts)
