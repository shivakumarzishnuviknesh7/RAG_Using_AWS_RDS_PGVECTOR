# frontend/chat_app.py  (you can keep the old file as backup if you want)
import os
import uuid
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
HEALTH_URL = f"{API_URL}/health"

CHAT_SEND_URL   = f"{API_URL}/chat/send"
CHAT_NEW_URL    = f"{API_URL}/chat/new"
CHAT_RESET_URL  = f"{API_URL}/chat/reset"
CONVS_URL       = f"{API_URL}/conversations"
HISTORY_URL     = f"{API_URL}/chat/history"

st.set_page_config(page_title="RDS pgvector · Live Chat", page_icon="💬", layout="wide")
st.title("💬 RDS pgvector · Live Chat (row-per-window RAG)")

# ------------------- helpers -------------------
def _get_json(url: str, params=None, timeout=15):
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _post_json(url: str, payload: dict, timeout=60):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _list_conversations(user_id: str):
    try:
        return _get_json(CONVS_URL, params={"user_id": user_id, "limit": 100})
    except Exception as e:
        st.error(f"Failed to load conversations: {e}")
        return []

def _load_history(user_id: str, conversation_id: str, limit: int = 500):
    try:
        return _get_json(HISTORY_URL, params={"user_id": user_id, "conversation_id": conversation_id, "limit": limit})
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return []

def _chat_send(user_id: str, conversation_id: str, content: str, test_group: int = 1):
    return _post_json(CHAT_SEND_URL, {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "content": content,
        "test_group": test_group,  # 0 = no RAG, 1 = with RAG
    }, timeout=120)

def _chat_new(user_id: str) -> str:
    try:
        resp = _post_json(CHAT_NEW_URL, {"user_id": user_id})
        return resp["conversation_id"]
    except Exception:
        # fallback client-generated id if server endpoint not available
        return f"c_{uuid.uuid4().hex[:12]}"

def _chat_reset(user_id: str, conversation_id: str):
    return _post_json(CHAT_RESET_URL, {"user_id": user_id, "conversation_id": conversation_id})

# ------------------- sidebar -------------------
with st.sidebar:
    st.header("Session")
    user_id = st.text_input("User ID", value=st.session_state.get("user_id", "u_demo"))
    st.session_state["user_id"] = user_id

    test_group = st.selectbox(
        "Test group",
        options=[1, 0],
        format_func=lambda x: "1 = RAG (on)" if x == 1 else "0 = No RAG (off)",
        index=0 if st.session_state.get("test_group", 1) == 1 else 1,
    )
    st.session_state["test_group"] = test_group

    cols = st.columns(2)
    with cols[0]:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state["_refresh_convs"] = True
    with cols[1]:
        if st.button("🔎 Health", use_container_width=True):
            try:
                st.json(requests.get(HEALTH_URL, timeout=10).json())
            except Exception as e:
                st.error(str(e))

    # Load conversations
    if user_id.strip():
        if st.session_state.get("_refresh_convs", True):
            st.session_state["convs"] = _list_conversations(user_id)
            st.session_state["_refresh_convs"] = False
        convs = st.session_state.get("convs", [])
    else:
        convs = []

    st.markdown("### Conversations")
    current_cid = st.session_state.get("conversation_id")
    for c in convs[:50]:
        label = f"{c['conversation_id']} · {c['turn_count']} turns"
        if st.button(label, key=f"sel_{c['conversation_id']}", use_container_width=True):
            st.session_state["conversation_id"] = c["conversation_id"]
            st.session_state["_force_history_reload"] = True
            st.rerun()

    st.markdown("---")
    if st.button("➕ New Chat", use_container_width=True):
        if not user_id.strip():
            st.warning("Enter a User ID first.")
        else:
            new_id = _chat_new(user_id)
            st.session_state["conversation_id"] = new_id
            st.success(f"Created: {new_id}")
            st.session_state["_force_history_reload"] = True
            st.session_state["_refresh_convs"] = True
            st.rerun()

    if current_cid:
        if st.button("🗑️ Reset current chat", use_container_width=True, type="secondary"):
            try:
                _chat_reset(user_id, current_cid)
                st.success("Cleared.")
                st.session_state["_refresh_convs"] = True
                st.session_state["_force_history_reload"] = True
                # keep same conversation id but empty history
            except Exception as e:
                st.error(str(e))

    st.caption(f"Backend: `{API_URL}`")

# ------------------- main: chat area -------------------
cid = st.session_state.get("conversation_id")
if not user_id.strip():
    st.info("Enter a **User ID** to start.")
    st.stop()

if not cid:
    st.info("Select a conversation on the left or click **New Chat**.")
    st.stop()

st.caption(f"User: **{user_id}** · Conversation: **{cid}** · Test group: **{test_group}**")

# Load history from backend (source of truth)
if st.session_state.get("_force_history_reload", True):
    history = _load_history(user_id, cid, limit=500)
    st.session_state["history"] = history
    st.session_state["_force_history_reload"] = False
else:
    history = st.session_state.get("history", [])

# Render history (Streamlit's chat bubbles)
for m in history:
    role = m.get("role")
    content = m.get("content", "")
    if role in ("user", "assistant"):
        with st.chat_message(role):
            st.markdown(content)

# Chat input
msg = st.chat_input("Type a message…")
if msg:
    # optimistic render user's message
    with st.chat_message("user"):
        st.markdown(msg)

    # send to backend
    try:
        resp = _chat_send(user_id, cid, msg, test_group=test_group)
        answer = resp.get("answer", "")
        snippets = resp.get("snippets", [])
    except Exception as e:
        answer = f"⚠️ Error: {e}"
        snippets = []

    # render assistant reply
    with st.chat_message("assistant"):
        st.markdown(answer)
        if snippets:
            with st.expander("🔎 Retrieved windows"):
                for i, s in enumerate(snippets, 1):
                    score = s.get("score")
                    try:
                        score = f"{float(score):.3f}"
                    except Exception:
                        score = str(score)
                    st.markdown(f"**[{i}]** {s.get('conversation_id')} · `{s.get('window_id')}` · score={score}")
                    st.write(s.get("text", ""))

    # refresh history so the left pane counts & ordering stay correct
    st.session_state["_force_history_reload"] = True
    st.session_state["_refresh_convs"] = True
    st.rerun()
