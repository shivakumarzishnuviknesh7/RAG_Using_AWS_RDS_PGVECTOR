# init_schema.py
from __future__ import annotations
from dotenv import load_dotenv
from db import db

# Load ..env file (DATABASE_URL must point to claradatabase now)
load_dotenv()

def main():
    conn = db()

    # Enable extensions (needed for our schema)
    conn.run("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.run("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    # Apply schema from db.sql
    with open("db.sql", "r", encoding="utf-8") as f:
        sql = f.read()
    conn.run(sql)

    # Seed one demo user (so FK constraints are happy)
    conn.run("INSERT INTO users (user_id) VALUES ('u_demo') ON CONFLICT (user_id) DO NOTHING;")

    print(" Schema applied + demo user added")

if __name__ == "__main__":
    main()
