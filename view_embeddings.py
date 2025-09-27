# view_embeddings.py
from __future__ import annotations
import math
from dotenv import load_dotenv
from db import db

load_dotenv()

def parse_vec(text: str) -> list[float]:
    # embedding::text looks like "[0.01, -0.02, ...]"
    t = text.strip()[1:-1]
    if not t:
        return []
    return [float(x) for x in t.split(",")]

rows = db().run("""
  SELECT window_id, user_id, conversation_id, embedding::text AS emb_txt
  FROM conv_windows
  WHERE embedding IS NOT NULL
  ORDER BY created_at DESC
  LIMIT 5
""")

print(f"\nFound {len(rows)} embedded window(s):")
for wid, uid, cid, emb_txt in rows:
    vec = parse_vec(emb_txt)
    dim = len(vec)
    # L2 norm just for sanity
    norm = math.sqrt(sum(v*v for v in vec)) if vec else 0.0
    preview = ", ".join(f"{v:.4f}" for v in vec[:8])
    print(f"- {wid} · {uid} · {cid} · dim={dim} · ||v||={norm:.3f} · [{preview}, ...]")

print("\n(done)")
