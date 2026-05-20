import streamlit as st
from chatbot_backend import chatbot

# Page Config 
st.set_page_config(
    page_title="NexaCorp Chatbot",
    layout="wide"
)

# Sidebar 
with st.sidebar:
    st.title("NexaCorp Assistant")
    st.divider()
    st.subheader("Chat History")

# Main UI
st.title("NexaCorp Global")
st.caption("Ask your HR-related questions here")

# Display chat history
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input 
prompt = st.chat_input(
    "Type here..."
)
