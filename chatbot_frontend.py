import streamlit as st

# Sidebar
with st.sidebar:
    st.title("NexaCorp Chatbot")
    st.subheader("Chat History")

# Main section
st.title("NexaCorp Global")
st.subheader("Ask your HR queries here")

user_query = st.chat_input("Type your question...")

if user_query:
    st.chat_message("user").write(user_query)
    st.chat_message("assistant").write(
        f"You asked: {user_query}"
    )