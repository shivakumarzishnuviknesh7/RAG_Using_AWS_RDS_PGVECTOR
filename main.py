# main.py
from __future__ import annotations
from typing import Dict, Any, List
import hashlib
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Body, Query

from models import IngestRequest, AskRequest, AskResponse, Snippet
from windowing import (
    extract_turn_texts,
    extract_turn_times,
    build_windows,
    window_time_bounds,
    tail_windows_for_new_turn,
)
from db import db
from rag_core import (
    embed_text,
    to_pgvector,
    retrieve_windows_vector,
    retrieve_windows_hybrid,
    build_prompt,
    chat,
)

app = FastAPI(title="RDS pgvector RAG (row-per-window)", version="0.4.0")


# ---------- helpers ----------
def _ensure_user(user_id: str):
    sql = """
    INSERT INTO users (user_id) VALUES (:uid)
    ON CONFLICT (user_id) DO NOTHING;
    """
    db().run(sql, uid=user_id)


def _sha256_norm(text: str) -> str:
    t = " ".join((text or "").split())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def _normalize_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cast DB-native types to JSON-friendly ones (UUID->str, Decimal->float)."""
    out: List[Dict[str, Any]] = []
    for h in (hits or []):
        d = dict(h)
        wid = d.get("window_id")
        if wid is not None and not isinstance(wid, str):
            d["window_id"] = str(wid)
        sc = d.get("score")
        if sc is not None and not isinstance(sc, (int, float)):
            try:
                d["score"] = float(sc)
            except Exception:
                d["score"] = None
        out.append(d)
    return out


def _insert_window_row(row: Dict[str, Any]):
    sql = """
    INSERT INTO conv_windows
      (user_id, conversation_id, start_index, end_index, turn_count,
       text, test_group, first_turn_at, last_turn_at, text_hash)
    VALUES
      (:uid, :cid, :si, :ei, :tc,
       :txt, :tg, :fta, :lta, :th)
    ON CONFLICT (user_id, conversation_id, start_index, end_index) DO NOTHING;
    """
    db().run(
        sql,
        uid=row["user_id"],
        cid=row["conversation_id"],
        si=row["start_index"],
        ei=row["end_index"],
        tc=row["turn_count"],
        txt=row["text"],
        tg=row["test_group"],
        fta=row["first_turn_at"],
        lta=row["last_turn_at"],
        th=row["text_hash"],
    )


def _log_event(user_id: str, test_group: int, event_name: str, data: Dict[str, Any]):
    try:
        db().run(
            """
            INSERT INTO analytics_events (user_id, test_group, event_name, data)
            VALUES (:uid, :tg, :ev, :data)
            """,
            uid=user_id, tg=test_group, ev=event_name, data=data,
        )
    except Exception:
        pass


# ---------- LIVE CHAT support ----------
def _next_turn_index(user_id: str, conversation_id: str) -> int:
    row = db().run(
        "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM conv_turns WHERE user_id=:u AND conversation_id=:c",
        u=user_id, c=conversation_id,
    )
    return int(row[0][0])


def _insert_turn(user_id: str, conversation_id: str, role: str, content: str, turn_index: int):
    db().run(
        """
        INSERT INTO conv_turns (user_id, conversation_id, turn_index, role, content)
        VALUES (:u, :c, :i, :r, :t)
        """,
        u=user_id, c=conversation_id, i=turn_index, r=role, t=content
    )


def _recent_user_assistant_texts(user_id: str, conversation_id: str, last_n: int = 200) -> List[str]:
    rows = db().run(
        """
        SELECT role, content
        FROM conv_turns
        WHERE user_id=:u AND conversation_id=:c AND role IN ('user','assistant')
        ORDER BY turn_index
        """,
        u=user_id, c=conversation_id,
    )
    turns = [{"role": r[0], "content": r[1]} for r in rows[-last_n:]]
    return extract_turn_texts(turns)


def _insert_tail_windows(user_id: str, conversation_id: str, texts: List[str], test_group: int):
    for si, ei, joined in tail_windows_for_new_turn(texts, min_len=2, max_len=4):
        db().run(
            """
            INSERT INTO conv_windows
              (user_id, conversation_id, start_index, end_index, turn_count, text, test_group, text_hash)
            VALUES
              (:u, :c, :si, :ei, :tc, :txt, :tg, :th)
            ON CONFLICT (user_id, conversation_id, start_index, end_index) DO NOTHING
            """,
            u=user_id, c=conversation_id, si=si, ei=ei, tc=(ei - si + 1),
            txt=joined, tg=test_group, th=_sha256_norm(joined)
        )


# ---------- routes ----------
@app.get("/health")
def health():
    try:
        db().run("SELECT 1")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ----- Conversations utilities -----
@app.get("/conversations")
def list_conversations(user_id: str = Query(..., min_length=1), limit: int = 50):
    rows = db().run(
        """
        SELECT conversation_id,
               COUNT(*) AS turn_count,
               MAX(created_at) AS last_at
        FROM conv_turns
        WHERE user_id = :u
        GROUP BY conversation_id
        ORDER BY last_at DESC
        LIMIT :n
        """,
        u=user_id, n=limit
    )
    return [
        {"conversation_id": r[0], "turn_count": int(r[1]), "last_at": str(r[2])}
        for r in rows
    ]


@app.get("/chat/history")
def chat_history(
    user_id: str = Query(..., min_length=1),
    conversation_id: str = Query(..., min_length=1),
    limit: int = 500,
):
    rows = db().run(
        """
        SELECT role, content, turn_index, created_at
        FROM conv_turns
        WHERE user_id = :u AND conversation_id = :c
        ORDER BY turn_index
        LIMIT :n
        """,
        u=user_id, c=conversation_id, n=limit
    )
    return [
        {"role": r[0], "content": r[1], "turn_index": int(r[2]), "created_at": str(r[3])}
        for r in rows
    ]


@app.post("/chat/new")
def chat_new(user_id: str = Body(..., embed=True)):
    _ensure_user(user_id)
    cid = f"c_{uuid4().hex[:12]}"
    return {"conversation_id": cid}


@app.post("/chat/reset")
def chat_reset(
    user_id: str = Body(..., embed=True),
    conversation_id: str = Body(..., embed=True),
):
    db().run(
        "DELETE FROM conv_turns WHERE user_id = :u AND conversation_id = :c",
        u=user_id, c=conversation_id
    )
    db().run(
        "DELETE FROM conv_windows WHERE user_id = :u AND conversation_id = :c",
        u=user_id, c=conversation_id
    )
    return {"ok": True}


# ----- Ingest / Ask -----
@app.post("/ingest")
def ingest(req: IngestRequest):
    _ensure_user(req.user_id)
    turns_dicts = [t.model_dump() for t in req.turns]
    texts = extract_turn_texts(turns_dicts)
    if not texts:
        raise HTTPException(status_code=400, detail="No user/assistant content to ingest.")

    times = extract_turn_times(turns_dicts)
    windows = build_windows(texts, min_len=2, max_len=4)

    inserted = 0
    for (start_idx, end_idx, joined_text) in windows:
        first_at, last_at = window_time_bounds(times, start_idx, end_idx)
        row = {
            "user_id": req.user_id,
            "conversation_id": req.conversation_id,
            "start_index": start_idx,
            "end_index": end_idx,
            "turn_count": (end_idx - start_idx + 1),
            "text": joined_text,
            "test_group": req.test_group,
            "first_turn_at": first_at,
            "last_turn_at": last_at,
            "text_hash": _sha256_norm(joined_text),
        }
        _insert_window_row(row)
        inserted += 1

    _log_event(
        user_id=req.user_id,
        test_group=req.test_group,
        event_name="INGEST_WINDOWS",
        data={"conversation_id": req.conversation_id, "windows": inserted},
    )
    return {"inserted": inserted}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        q_vec = embed_text(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    qvec_literal = to_pgvector(q_vec)
    try:
        if req.hybrid:
            hits = retrieve_windows_hybrid(
                user_id=req.user_id, query_text=req.question,
                qvec=qvec_literal, top_k=req.top_k,
            )
        else:
            hits = retrieve_windows_vector(
                user_id=req.user_id, qvec=qvec_literal, top_k=req.top_k,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    try:
        messages = build_prompt(req.question, hits)
        answer = chat(messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    _log_event(
        user_id=req.user_id, test_group=req.test_group,
        event_name="RAG_ANSWER",
        data={"top_k": req.top_k, "hybrid": req.hybrid, "hits": [h.get("window_id") for h in hits]},
    )

    norm_hits = _normalize_hits(hits)
    return AskResponse(answer=answer, snippets=[Snippet(**h) for h in norm_hits])


# ----- Live chat -----
@app.post("/chat/send", response_model=AskResponse)
def chat_send(
    user_id: str = Body(...),
    conversation_id: str = Body(...),
    content: str = Body(...),
    test_group: int = Body(1, embed=True),
    hybrid: bool = Body(False, embed=True),
):
    _ensure_user(user_id)

    # 1) append user's turn
    idx = _next_turn_index(user_id, conversation_id)
    _insert_turn(user_id, conversation_id, "user", content, idx)

    # 2) create windows ending at this user turn
    texts = _recent_user_assistant_texts(user_id, conversation_id)
    _insert_tail_windows(user_id, conversation_id, texts, test_group)

    # 3) answer (vector or hybrid RAG)
    if test_group == 0:
        messages = [
            {"role": "system", "content": "Be concise and helpful."},
            {"role": "user", "content": content},
        ]
        answer = chat(messages)
        hits: List[Dict[str, Any]] = []
    else:
        q_vec = embed_text(content)
        qvec_literal = to_pgvector(q_vec)
        if hybrid:
            hits = retrieve_windows_hybrid(user_id=user_id, query_text=content, qvec=qvec_literal, top_k=6)
        else:
            hits = retrieve_windows_vector(user_id=user_id, qvec=qvec_literal, top_k=6)
        messages = build_prompt(content, hits)
        answer = chat(messages)

    # 4) append assistant reply + windowize again
    idx2 = _next_turn_index(user_id, conversation_id)
    _insert_turn(user_id, conversation_id, "assistant", answer, idx2)
    texts2 = _recent_user_assistant_texts(user_id, conversation_id)
    _insert_tail_windows(user_id, conversation_id, texts2, test_group)

    # 5) return
    norm_hits = _normalize_hits(hits)
    return AskResponse(answer=answer, snippets=[Snippet(**h) for h in norm_hits])
