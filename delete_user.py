# delete_user.py
from __future__ import annotations
import argparse
from dotenv import load_dotenv
from db import db

load_dotenv()

def user_exists(user_id: str) -> bool:
    row = db().run("SELECT 1 FROM users WHERE user_id = :u", u=user_id)
    return bool(row)

def delete_user(user_id: str, delete_analytics: bool = True):
    if not user_exists(user_id):
        return {"ok": False, "message": f"user_id '{user_id}' not found"}

    # cascade will wipe conv_turns + conv_windows
    db().run("DELETE FROM users WHERE user_id = :u", u=user_id)

    # optional: analytics has no FK, clean those up separately
    if delete_analytics:
        db().run("DELETE FROM analytics_events WHERE user_id = :u", u=user_id)

    return {"ok": True, "message": f"deleted user '{user_id}' and all related data"}

def main():
    ap = argparse.ArgumentParser(description="Delete an entire user (cascades turns + windows/embeddings).")
    ap.add_argument("--user", required=True, help="user_id")
    ap.add_argument("--keep-analytics", action="store_true", help="do not delete analytics rows")
    args = ap.parse_args()

    res = delete_user(args.user, delete_analytics=(not args.keep_analytics))
    print(res)

if __name__ == "__main__":
    main()
