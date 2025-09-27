from __future__ import annotations

import os, urllib.parse as up, ssl, pg8000.native as pg
import certifi
from dotenv import load_dotenv

# Load ..env when run from scripts/tests
load_dotenv()

_conn: pg.Connection | None = None

def _parse_db_url(db_url: str) -> dict:
    u = up.urlparse(db_url)
    if u.scheme not in ("postgres", "postgresql"):
        raise ValueError("DATABASE_URL must start with postgresql://")

    # Base trust store from certifi
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    # Optionally add AWS RDS bundle (recommended)
    ca_bundle = os.getenv("RDS_CA_BUNDLE")
    if ca_bundle and os.path.exists(ca_bundle):
        try:
            ssl_ctx.load_verify_locations(cafile=ca_bundle)
        except Exception as e:
            print(f"[db] Warning: couldn't load RDS_CA_BUNDLE ({ca_bundle}): {e}")

    # Optional dev override (only if you absolutely must)
    if os.getenv("ALLOW_INSECURE_SSL", "").lower() in ("1", "true", "yes"):
        print("[db] WARNING: ALLOW_INSECURE_SSL is enabled (dev only).")
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    return dict(
        user=up.unquote(u.username or ""),
        password=up.unquote(u.password or ""),
        host=u.hostname or "localhost",
        port=u.port or 5432,
        database=(u.path or "/").lstrip("/"),
        ssl_context=ssl_ctx,
    )

def db() -> pg.Connection:
    global _conn
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in environment")
    if _conn is None:
        _conn = pg.Connection(**_parse_db_url(db_url))
    else:
        try:
            _conn.run("SELECT 1")
        except Exception:
            _conn = pg.Connection(**_parse_db_url(db_url))
    return _conn
