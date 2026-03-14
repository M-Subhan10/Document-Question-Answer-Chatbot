"""
api.py — Local REST API for docqa.

Run locally:
  python api.py

Or:
  uvicorn api:app --host 127.0.0.1 --port 7861
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import uvicorn

from config import API_PORT, SERVER_HOST
from service import SERVICE


class QueryRequest(BaseModel):
    session_id: str
    question: str


app = FastAPI(
    title="docqa local API",
    version="1.0.0",
    description="Single-document ingest and query API for local testing.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return SERVICE.ingest_bytes(file.filename, data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/query")
def query(payload: QueryRequest) -> dict:
    try:
        return SERVICE.query_session(payload.session_id, payload.question)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}")
def session_info(session_id: str) -> dict:
    try:
        return SERVICE.get_session_info(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    deleted = SERVICE.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return {"deleted": True, "session_id": session_id}


if __name__ == "__main__":
    uvicorn.run("api:app", host=SERVER_HOST, port=API_PORT, reload=False)
