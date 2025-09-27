from __future__ import annotations

import os, urllib.parse as up
from dotenv import load_dotenv
from db import db, _parse_db_url  # we’ll reuse your parser

load_dotenv()

def make_admin_conn_to_postgres():
    """Connect to the default 'postgres' database regardless of the DB in DATABASE_URL."""
    db_url = os.environ["DATABASE_URL"]
    parts = _parse_db_url(db_url)
    # force database to 'postgres' so we can create others
    parts["database"] = "postgres"
    import pg8000.native as pg
    return pg.Connection(**parts)

def main():
    try:
        admin = make_admin_conn_to_postgres()
        # CREATE DATABASE is not IN-transaction in Postgres, pg8000 handles autocommit for DDL
        admin.run("CREATE DATABASE claradatabase;")
        print(" created DB 'claradatabase'")
    except Exception as e:
        # If it already exists, Postgres throws 42P04
        msg = str(e)
        if "42P04" in msg or "already exists" in msg:
            print("ℹ DB 'claradatabase' already exists.")
        else:
            raise

if __name__ == "__main__":
    main()
