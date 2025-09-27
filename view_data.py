# view_data.py
from db import db

def show_users():
    rows = db().run("SELECT * FROM users;")
    for r in rows:
        print(r)

def show_turns():
    rows = db().run("""
        SELECT user_id, conversation_id, turn_index, role, content
        FROM conv_turns
        ORDER BY created_at DESC
        ;
    """)
    for r in rows:
        print(r)

def show_windows():
    rows = db().run("""
        SELECT window_id, user_id, conversation_id, start_index, end_index, text, embedding IS NOT NULL
        FROM conv_windows
        ORDER BY created_at DESC
        LIMIT 10;
    """)
    for r in rows:
        print(r)

if __name__ == "__main__":
    print("=== Users ===")
    show_users()
    print("\n=== Recent Turns ===")
    show_turns()
    print("\n=== Recent Windows ===")
    show_windows()
