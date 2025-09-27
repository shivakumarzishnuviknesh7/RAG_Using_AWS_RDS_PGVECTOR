
import os
from db import db  # uses the db.py we wrote

# Make sure ..env is loaded (python-dotenv)
from dotenv import load_dotenv
load_dotenv()

try:
    conn = db()
    rows = conn.run("SELECT version();")
    print(" Connected! Postgres version:", rows[0][0])
except Exception as e:
    print(" Connection failed:", e)
