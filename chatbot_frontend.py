import streamlit as st

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

# User Input 
prompt = st.chat_input(
    "Type here..."
)
