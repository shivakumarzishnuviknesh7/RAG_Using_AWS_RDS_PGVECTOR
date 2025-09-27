# rag_core.py
from __future__ import annotations
import os
import re
import random
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from db import db

# Load .env at import time so OPENAI_API_KEY / EMBED_MODEL / CHAT_MODEL are available
load_dotenv()

# ---- Models / .env ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gpt-4o-mini")

# Hybrid weights (tunable via .env)
W_VECTOR = float(os.getenv("HYBRID_W_VECTOR", "0.7"))
W_FTS    = float(os.getenv("HYBRID_W_FTS", "0.3"))

# Vector time decay (days), applies to both vector-only and hybrid scoring
DECAY_DAYS = float(os.getenv("RAG_DECAY_DAYS", "45.0"))

# Candidate sizes for late fusion (scan more than top_k to let the re-ranker work)
VEC_CAND_MULT = int(os.getenv("HYBRID_VEC_CAND_MULT", "8"))   # vector candidates = top_k * this
FTS_CAND_MULT = int(os.getenv("HYBRID_FTS_CAND_MULT", "8"))   # keyword candidates = top_k * this
MIN_CANDS     = int(os.getenv("HYBRID_MIN_CANDS", "50"))      # floor for candidate set sizes


# ---- OpenAI client (lazy) ----
def _get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


# ---- Lightweight intent guards (prevents needless retrieval + hallucinations) ----
ACK_ONLY = re.compile(r"^(ok(ay)?|thanks?|thank\s+you|thx|ty|👍+|👎+|👌+|🙏+|\.)+$", re.I)
YESNO_ONLY = re.compile(r"^(yes|yeah|yep|yup|correct|right|no|nope)\.?$", re.I)
GREETING_ONLY = re.compile(r"^(hi|hello|hey|hallo|hola|namaste)\b[!.]*$", re.I)

# Fact statements like: "its dark blue cup", "it's green", "my bag is green",
# "I am 68", "I'm Anna", "I like chess", "I have two cats"
FACT_PATTERN = re.compile(r"^(it'?s|its|my|i(?:'m| am| have| like))\b.+", re.I)

ACK_REPLIES = [
    "You’re welcome! 😊",
    "Glad I could help!",
    "Anytime—happy to help.",
    "Okay!",
]
YES_REPLIES = ["Okay!", "Got it.", "Thanks for confirming."]
NO_REPLIES  = ["Okay, no problem.", "Got it, thanks for telling me."]


def is_acknowledgment(text: str) -> bool:
    return bool(ACK_ONLY.match((text or "").strip()))


def is_yesno(text: str) -> str | None:
    """Return 'yes'/'no' if message is a bare yes/no; else None."""
    t = (text or "").strip().lower().strip(".")
    if YESNO_ONLY.match(t):
        return "no" if t in {"no", "nope"} else "yes"
    return None


def is_greeting(text: str) -> bool:
    return bool(GREETING_ONLY.match((text or "").strip()))


def is_fact_statement(text: str) -> bool:
    return bool(FACT_PATTERN.match((text or "").strip()))


def quick_ack() -> str:
    return random.choice(ACK_REPLIES)


def quick_yes() -> str:
    return random.choice(YES_REPLIES)


def quick_no() -> str:
    return random.choice(NO_REPLIES)


def confirm_fact(text: str) -> str:
    """Return a short confirmation echo without adding unrelated info."""
    s = (text or "").strip()
    # Normalize leading "its" -> "it's"
    s = re.sub(r"^its\b", "it's", s, flags=re.I)
    # Ensure single ending period
    s = s.rstrip(".! ")
    return f"Got it, {s}."


# ---- Helpers ----
def to_pgvector(vec: List[float]) -> str:
    """Format a Python list[float] into pgvector literal: [0.1,0.2,...]."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def embed_text(text: str) -> List[float]:
    """Get a single embedding vector."""
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


# ---- Retrieval (row-per-window) ----
def retrieve_windows_vector(
    *,
    user_id: str,
    qvec: str,
    top_k: int = 15,
    decay_days: float = DECAY_DAYS,
) -> List[Dict[str, Any]]:
    """
    Vector-only retrieval from conv_windows for a given user.
    Score = cosine_sim * time_decay(age_days, decay_days).
    """
    sql = """
    WITH s AS (
      SELECT
        window_id,
        conversation_id,
        text,
        1 - (embedding <=> :qv::vector) AS sim,
        EXTRACT(EPOCH FROM (now() - COALESCE(last_turn_at, created_at)))/86400.0 AS age_days
      FROM conv_windows
      WHERE user_id = :uid
        AND embedding IS NOT NULL
    )
    SELECT window_id, conversation_id, text,
           (sim * EXP(-GREATEST(age_days,0)/:decay)) AS score
    FROM s
    ORDER BY score DESC
    LIMIT :k;
    """
    rows = db().run(sql, qv=qvec, uid=user_id, decay=decay_days, k=top_k)
    return [
        {"window_id": r[0], "conversation_id": r[1], "text": r[2], "score": float(r[3])}
        for r in rows
    ]


def retrieve_windows_hybrid(
    *,
    user_id: str,
    query_text: str,
    qvec: str,
    top_k: int = 15,
    vec_weight: float = W_VECTOR,
    fts_weight: float = W_FTS,
    decay_days: float = DECAY_DAYS,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval = late fusion of:
      - vector similarity (cosine) with time decay, and
      - keyword full-text rank (ts_rank on 'fts').

    Approach:
      1) Pull top-N vector candidates and top-N FTS candidates (N >> top_k).
      2) Union + dedupe on window_id.
      3) Rescale both scores to ~[0,1] within the candidate set.
      4) Combine: vec_weight * vscore + fts_weight * kscore.
      5) Return top_k.

    Notes:
      - Uses 'simple' config to match your fts column creation.
      - websearch_to_tsquery handles quoted phrases, AND/OR, etc.
    """
    vec_lim = max(top_k * VEC_CAND_MULT, MIN_CANDS)
    fts_lim = max(top_k * FTS_CAND_MULT, MIN_CANDS)

    sql = """
    WITH
    vec AS (
      SELECT
        window_id,
        conversation_id,
        text,
        1 - (embedding <=> :qv::vector) AS vec_sim,
        EXTRACT(EPOCH FROM (now() - COALESCE(last_turn_at, created_at)))/86400.0 AS age_days
      FROM conv_windows
      WHERE user_id = :uid
        AND embedding IS NOT NULL
      ORDER BY (embedding <=> :qv::vector) ASC
      LIMIT :vlim
    ),
    fts AS (
      SELECT
        window_id,
        conversation_id,
        text,
        ts_rank(fts, websearch_to_tsquery('simple', :qtxt)) AS fts_rank
      FROM conv_windows
      WHERE user_id = :uid
        AND fts @@ websearch_to_tsquery('simple', :qtxt)
      ORDER BY fts_rank DESC
      LIMIT :flim
    ),
    cand AS (
      SELECT v.window_id,
             v.conversation_id,
             v.text,
             v.vec_sim,
             v.age_days,
             0::float AS fts_rank
      FROM vec v
      UNION
      SELECT f.window_id,
             f.conversation_id,
             f.text,
             0::float AS vec_sim,
             EXTRACT(EPOCH FROM (now() - COALESCE(NULLIF(NULL, NULL), now())))/86400.0 AS age_days_dummy, -- keep shape
             f.fts_rank
      FROM fts f
    ),
    merged AS (
      -- Aggregate to combine vec/fts for the same window_id
      SELECT
        window_id,
        ANY_VALUE(conversation_id) AS conversation_id,
        ANY_VALUE(text)           AS text,
        MAX(vec_sim)              AS vec_sim,
        MAX(fts_rank)             AS fts_rank,
        -- prefer a real age_days if present (0 if unknown)
        MAX(age_days) FILTER (WHERE age_days IS NOT NULL) AS age_days
      FROM cand
      GROUP BY window_id
    ),
    scaled AS (
      -- Normalize to ~[0,1] to make weights meaningful
      SELECT
        window_id,
        conversation_id,
        text,
        CASE
          WHEN MAX(vec_sim) OVER () > 0 THEN vec_sim / MAX(vec_sim) OVER ()
          ELSE 0
        END AS vscore_raw,
        CASE
          WHEN MAX(fts_rank) OVER () > 0 THEN fts_rank / MAX(fts_rank) OVER ()
          ELSE 0
        END AS kscore_raw,
        COALESCE(age_days, 0) AS age_days
      FROM merged
    ),
    scored AS (
      SELECT
        window_id,
        conversation_id,
        text,
        -- Apply exponential time decay to the vector part only (keeps "freshness" effect)
        (:vw * (vscore_raw * EXP(-GREATEST(age_days,0)/:decay))) +
        (:fw * kscore_raw) AS hybrid_score
      FROM scaled
    )
    SELECT window_id, conversation_id, text, hybrid_score
    FROM scored
    ORDER BY hybrid_score DESC
    LIMIT :k;
    """
    rows = db().run(
        sql,
        uid=user_id,
        qv=qvec,
        qtxt=query_text,
        vlim=vec_lim,
        flim=fts_lim,
        vw=vec_weight,
        fw=fts_weight,
        decay=decay_days,
        k=top_k,
    )
    return [
        {"window_id": r[0], "conversation_id": r[1], "text": r[2], "score": float(r[3])}
        for r in rows
    ]


# ---- Prompt + Chat ----
def build_prompt(question: str, snippets: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Supportive, memory-aware prompt designed for older adults.
    - If context exists: answer briefly using it.
    - If context is empty/insufficient: ask exactly ONE gentle follow-up and invite sharing so we can remember.
    - Keep replies short (1–3 short sentences), warm, and simple.
    """
    context = "\n\n---\n\n".join(s["text"] for s in snippets).strip()
    has_context = bool(context)

    system = (
        "Your name is  CLARA, a warm and supportive assistant for older adults.\n"
        "You are a helpful assistant designed to support individuals with language and cognitive tasks, specifically for word finding and word description tasks. Your role is to offer simple, clear, and helpful responses that assist the user in identifying the correct word or providing descriptive information when they are unable to find the word. \n"
        "Write clearly, in short sentences, with simple words. Be kind and encouraging.\n\n"

        "Rules:\n"
        "1) If useful context is provided, use it directly. If several snippets relate, connect them briefly.\n"
        "2) If context is missing or not enough, do NOT say only “I don’t know.”\n"
        "   Instead: (a) say you don’t have it in memory yet, (b) ask exactly ONE gentle follow-up,\n"
        "   and (c) invite the user to share a small detail so you can remember it next time.\n"
        "3) If the user shares a personal fact (e.g., about a car, pet, medication), acknowledge it politely\n"
        "   and offer to remember it. Keep confirmations short and never list many facts unless asked.\n"
        "4) If the user only says 'ok/thanks/yes/no', reply with a very brief, polite line (no new topics).\n"
        "5) Never invent details that aren’t in the context or the user’s latest message.\n"
        "6) Keep responses to 1–5 sentences. Avoid long lists. Keep tone calm and positive.\n"
    )

    # Let the model see whether we actually have anything to work with.
    # This enables the “no memory yet — ask one gentle follow-up” behavior purely via prompt.
    user = (
        f"User message:\n{question}\n\n"
        f"Context from memory (may be empty):\n"
        f"{context if has_context else '(no matching context)'}\n\n"
        "Your task:\n"
        "- If the context contains the answer, give a short grounded reply using it.\n"
        "- If the context is empty or insufficient, \n"
        "  ask exactly ONE gentle follow-up question, and invite the user to share a small detail you can remember.\n"
        "- Keep it to 1–3 short sentences total."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]



def chat(messages: List[Dict[str, str]]) -> str:
    """Call the chat model."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0.2")),
        top_p=1,
        max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "512")),
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


# ---- Public entrypoint for your pipeline ----
def answer(user_id: str, query: str, mode: str = "hybrid") -> str:
    """
    Main entrypoint:
      - Short-circuits greetings, acknowledgments, yes/no, and fact statements.
      - Otherwise runs retrieval -> builds grounded prompt -> calls the chat model.
    """
    q = (query or "").strip()

    # 1) Bare greetings -> tiny reply
    if is_greeting(q):
        return "Hi! How can I help you today?"

    # 2) Bare acknowledgments -> tiny reply
    if is_acknowledgment(q):
        return quick_ack()

    # 3) Bare yes/no -> tiny reply (no new info)
    yn = is_yesno(q)
    if yn == "yes":
        return quick_yes()
    if yn == "no":
        return quick_no()

    # 4) Fact statements -> short confirmation (no retrieval)
    if is_fact_statement(q):
        return confirm_fact(q)

    # 5) Normal RAG flow
    qvec = to_pgvector(embed_text(q))

    if mode == "hybrid":
        snippets = retrieve_windows_hybrid(user_id=user_id, query_text=q, qvec=qvec, top_k=15)
    else:
        snippets = retrieve_windows_vector(user_id=user_id, qvec=qvec, top_k=15)

    messages = build_prompt(q, snippets)
    return chat(messages)
