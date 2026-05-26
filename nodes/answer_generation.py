from schemas.chatbot_state import ChatbotState
from config.llm import llm
from prompts.answer_prompt import ANSWER_PROMPT

def generate_answer(state: ChatbotState):
    query = state["query"]
    docs = state["reranked_docs"]
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = ANSWER_PROMPT.format(
        query=query,
        context=context
    )

    result = llm.invoke(prompt).content
    return {
    "final_answer": result,
    "sources": [
        {
            "section": doc.metadata.get("section"),
            "subsection": doc.metadata.get("subsection")
        }
        for doc in docs
    ]
    }