import streamlit as st
from bot_backend import get_bot_response

st.set_page_config(page_title="Kerala Cyber Law Assistant", layout="wide")
st.title("Kerala Cyber Law Assistant 🤖")
st.warning(
    "DISCLAIMER: I am not a lawyer. This is not legal advice. "
    "For urgent help, call 1930 for financial fraud or visit cybercrime.gov.in."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("About This Project")
    st.markdown("Hybrid Classifier-RAG bot for hyperlocal cybercrime help in Kerala.")
    st.markdown("**Team:** Person A, B, C, D")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot_reply = get_bot_response(user_query)
            st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

st.markdown("---")
st.caption("Built for Kerala Cyber Law Assistant — Frontend & QA (Person D)")
