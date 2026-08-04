import streamlit as st
import requests

st.set_page_config(page_title="Motor Pump Assistant", page_icon="⚙️")
st.title("Motor Pump Assistant")
st.write("Ask about the equipment manual, maintenance, or pump handling.")

if "history" not in st.session_state:
    st.session_state.history = []

# text input bound to session state so buttons can change it programmatically
query = st.text_input("Ask a question:", key="query")

col1, col2 = st.columns([1, 0.3])
with col1:
    if st.button("Send") and st.session_state.query:
        q = st.session_state.query
        with st.spinner("Thinking..."):
            try:
                resp = requests.post("http://127.0.0.1:8000/ask", json={"text": q}, timeout=30)
                data = resp.json()
            except Exception as exc:
                data = {"route": "error", "answer": f"Request failed: {exc}"}

        st.session_state.history.append({
            "question": q,
            "answer": data.get("answer", ""),
            "route": data.get("route", ""),
            "sources": data.get("sources", []),
            "preview": data.get("context_preview", ""),
        })
        # clear input after sending
        st.session_state.query = ""

with col2:
    if st.button("Clear"):
        st.session_state.history = []

st.markdown("---")

if not st.session_state.history:
    st.info("No conversations yet — ask a question to get started.")

for i, item in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Q:** {item['question']}")
    st.markdown(f"**A:** {item['answer']}  ")

    if item.get("sources"):
        st.markdown("**Sources:** " + ", ".join(item["sources"]))

    if item.get("preview"):
        with st.expander("Context preview"):
            st.write(item["preview"])

    # follow-up button prefills the input with a clarifying question
    follow_key = f"follow_{i}"
    if st.button("Ask a follow-up", key=follow_key):
        # prefill the input box with a polite follow-up starter
        st.session_state.query = f"Following up on: {item['question']} — could you clarify..."

    st.write("---")
