from schemas.query_schema import QueryAnalysis
from schemas.chatbot_state import ChatbotState
from config.llm import llm
from prompts.decomposition_prompt import DECOMPOSITION_PROMPT

def query_decomposition(state: ChatbotState):
    query = state["query"]
    prompt = DECOMPOSITION_PROMPT.format(
        query=query
    )

    structured_llm = llm.with_structured_output(QueryAnalysis)
    result = structured_llm.invoke(prompt)

    if result.needs_decomposition == "no":
        return {"needs_decomposition": "no", "subqueries": [query]}
    return {"needs_decomposition": "yes", "subqueries": result.subqueries}
    