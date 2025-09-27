# windowing.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import hashlib
from datetime import datetime

# A rare separator so the model can "see" boundaries between turns in a window
SEP = " ⟂ "


def _normalize_text(s: str) -> str:
    """Collapse whitespace/newlines and strip."""
    return " ".join((s or "").split())


def extract_turn_texts(turns: List[Dict]) -> List[str]:
    """
    Input: turns like [{"role":"assistant","content":"Hi"}, {"role":"user","content":"Hey"}]
    Output: ["Hi", "Hey"] (only user/assistant; normalized; empty removed)
    """
    keep = {"user", "assistant"}
    out: List[str] = []
    for m in (turns or []):
        if m.get("role") in keep and m.get("content"):
            txt = _normalize_text(m["content"])
            if txt:
                out.append(txt)
    return out


def extract_turn_times(turns: List[Dict]) -> List[Optional[datetime]]:
    """
    Extract per-turn timestamps if present as ISO strings (RFC3339/ISO 8601).
    If not present or parse fails, element is None.
    """
    times: List[Optional[datetime]] = []
    for m in (turns or []):
        ts = m.get("created_at")
        if not ts:
            times.append(None)
            continue
        try:
            # Allow 'Z' suffix or timezone-aware strings
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            times.append(t)
        except Exception:
            times.append(None)
    return times


def build_windows(
    turn_texts: List[str],
    min_len: int = 2,
    max_len: int = 4,
) -> List[Tuple[int, int, str]]:
    """
    Build overlapping windows from the sequence of turn texts.
    Returns a list of (start_index, end_index_inclusive, joined_text).

    Example:
      turn_texts = ["A","B","C","D"]
      windows (min_len=2,max_len=3) =>
        (0,1,"A ⟂ B")
        (0,2,"A ⟂ B ⟂ C")
        (1,2,"B ⟂ C")
        (1,3,"B ⟂ C ⟂ D")
        (2,3,"C ⟂ D")
    """
    n = len(turn_texts)
    out: List[Tuple[int, int, str]] = []
    for i in range(n):
        for L in range(min_len, max_len + 1):
            j = i + L  # exclusive end
            if j <= n:
                start = i
                end_inclusive = j - 1
                joined = SEP.join(turn_texts[i:j])
                out.append((start, end_inclusive, joined))
    return out


def tail_windows_for_new_turn(
    texts: List[str],
    min_len: int = 2,
    max_len: int = 4,
) -> List[Tuple[int, int, str]]:
    """
    Build only the windows that END at the most recent turn.
    Use this during live chat to avoid regenerating all windows on every message.

    Example:
      texts=["A","B","C","D"], min_len=2,max_len=3
      returns:
        (2,3,"C ⟂ D")
        (1,3,"B ⟂ C ⟂ D")
    """
    n = len(texts)
    if n == 0:
        return []
    out: List[Tuple[int, int, str]] = []
    for L in range(min_len, max_len + 1):
        i = max(0, n - L)
        j = n
        if j - i >= min_len:
            out.append((i, j - 1, SEP.join(texts[i:j])))
    return out


def window_time_bounds(
    per_turn_times: List[Optional[datetime]],
    start_index: int,
    end_index: int,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    From the per-turn timestamp list, return (first_turn_at, last_turn_at) for this window.
    If all are None, returns (None, None).
    """
    slice_times = [t for t in per_turn_times[start_index:end_index + 1] if t is not None]
    if not slice_times:
        return None, None
    return min(slice_times), max(slice_times)


def text_hash(text: str) -> str:
    """Deterministic hash for dedup (store in text_hash column if you wish)."""
    return hashlib.sha256((_normalize_text(text)).encode("utf-8")).hexdigest()


# Backwards-compatible alias used elsewhere
_sha256_norm = text_hash
