# embed_worker.py
from __future__ import annotations
import os
import time
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Load ..env so OPENAI_API_KEY / DATABASE_URL are available when running directly
load_dotenv()

# Flat imports because your files are in the project root
from db import db
from rag_core import to_pgvector

# ---------------- Config ----------------
EMBED_MODEL  = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE   = int(os.getenv("EMBED_BATCH_SIZE", "128"))
SLEEP_EMPTY  = float(os.getenv("WORKER_SLEEP_EMPTY", "2.0"))   # seconds between polls when no work
SLEEP_ERROR  = float(os.getenv("WORKER_SLEEP_ERROR", "5.0"))   # cool-off after API/DB errors
# ----------------------------------------

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def fetch_pending(limit: int) -> List[Tuple[str, str]]:
    """
    Return list of (window_id, text) for rows needing embeddings.
    """
    sql = """
    SELECT window_id, text
    FROM conv_windows
    WHERE embedding IS NULL
    ORDER BY created_at
    LIMIT :lim;
    """
    rows = db().run(sql, lim=limit)  # pass kwargs, not a dict
    return [(str(r[0]), r[1]) for r in rows]


def update_embedding(window_id: str, vec: List[float]) -> None:
    """
    Update a single row with the computed vector.
    """
    sql = """
    UPDATE conv_windows
    SET embedding = :v::vector
    WHERE window_id = :wid;
    """
    db().run(sql, v=to_pgvector(vec), wid=window_id)  # kwargs


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Call OpenAI Embeddings once with a batch of inputs.
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    print(f"[worker] starting — model={EMBED_MODEL}, batch_size={BATCH_SIZE}")
    while True:
        try:
            pending = fetch_pending(BATCH_SIZE)
            if not pending:
                time.sleep(SLEEP_EMPTY)
                continue

            ids, texts = zip(*pending)  # type: ignore

            # Try batch first
            try:
                embs = embed_batch(list(texts))
                for wid, vec in zip(ids, embs):
                    update_embedding(wid, vec)
                print(f"[worker] embedded {len(ids)} window(s)")
            except Exception as e:
                # Batch failed: back off to per-row (rate limits, rare content issues, etc.)
                print(f"[worker] batch error: {e}. Falling back to per-row.")
                for wid, txt in pending:
                    try:
                        vec = embed_batch([txt])[0]
                        update_embedding(wid, vec)
                    except Exception as ee:
                        # Skip problematic row; log and continue
                        print(f"[worker] row error window_id={wid}: {ee}")
                        time.sleep(0.25)
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("[worker] interrupted — exiting.")
            break
        except Exception as e:
            print(f"[worker] unexpected error: {e}")
            time.sleep(SLEEP_ERROR)


if __name__ == "__main__":
    # Env sanity checks
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set in environment")
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL not set in environment")
    main()
