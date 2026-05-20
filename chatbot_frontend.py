import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from chatbot_backend import (
    chatbot,
    retrieve_all_threads,
    thread_document_metadata
)

# ===================== Page Config =====================
st.set_page_config(
    page_title="NexaCorp Chatbot",
    layout="wide"
)

# ===================== Utility Functions =====================
def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def reset_chat():
    new_id = generate_thread_id()
    st.session_state.thread_id = new_id
    st.session_state.message_history = []
    add_thread(new_id)

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return state.values.get("messages", [])

# ===================== Session State =====================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = {}

add_thread(st.session_state.thread_id)

thread_key = str(st.session_state.thread_id)

thread_docs = st.session_state.ingested_docs.setdefault(
    thread_key,
    {}
)

threads = st.session_state.chat_threads[::-1]


# ===================== Sidebar =====================

with st.sidebar:

    st.title("🤖 NexaCorp Assistant")

    st.caption(
        f"Thread ID:\n`{thread_key[:8]}...`"
    )

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):
        reset_chat()
        st.rerun()

    st.divider()

    st.subheader("Past Chats")

    selected_thread = None

    for thread in threads:

        if st.button(
            thread[:18],
            key=thread
        ):
            selected_thread = thread


# ===================== Main UI =====================
st.title("NexaCorp Global")
st.caption(
    "Ask your HR-related questions here"
)

# Display chat history
for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===================== User Input =====================
prompt = st.chat_input(
    "Ask something..."
)
if prompt:
    st.session_state.message_history.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    CONFIG = {
        "configurable": {
            "thread_id": thread_key
        }
    }

    with st.chat_message("assistant"):

        status_box = {"box": None}

        def response_stream():

            for chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=prompt
                        )
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):

                if isinstance(
                    chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        chunk,
                        "name",
                        "tool"
                    )

                    if status_box["box"] is None:

                        status_box["box"] = st.status(
                            f"Using {tool_name}",
                            expanded=True
                        )

                    else:

                        status_box["box"].update(
                            label=f"Using {tool_name}",
                            state="running"
                        )

                elif isinstance(
                    chunk,
                    AIMessage
                ):

                    yield chunk.content


        response = st.write_stream(
            response_stream()
        )

        if status_box["box"]:

            status_box["box"].update(
                label="Completed",
                state="complete",
                expanded=False
            )

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    metadata = thread_document_metadata(
        thread_key
    )

    if metadata:

        st.caption(
            f"""
📄 Active Document:
{metadata['filename']}

Pages: {metadata['documents']}

Chunks: {metadata['chunks']}
"""
        )


# ===================== Thread Switching =====================

if selected_thread:

    st.session_state.thread_id = selected_thread

    loaded_messages = load_conversation(
        selected_thread
    )

    temp = []

    for msg in loaded_messages:

        if isinstance(
            msg,
            HumanMessage
        ):
            role = "user"
        else:
            role = "assistant"

        temp.append(
            {
                "role": role,
                "content": msg.content
            }
        )

    st.session_state.message_history = temp

    st.session_state.ingested_docs.setdefault(
        str(selected_thread),
        {}
    )

    st.rerun()