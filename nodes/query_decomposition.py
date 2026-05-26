from schemas.query_schema import QueryAnalysis
from schemas.chatbot_state import ChatbotState

def query_decomposition(state: ChatbotState):
    query = state["query"]
    decomposition_prompt = f"""You are a query analysis assistant for a RAG chatbot.
        Determine whether the user query should be decomposed into multiple subqueries.

        Decompose if:
        - the query contains multiple topics   
        - comparison queries
        - multi-hop reasoning
        - conjunctions like and/or/vs/compare

        If decomposition is needed:
        - set needs_decomposition = "yes"
        - generate standalone retrieval-friendly subqueries

        If not needed:
        - set needs_decomposition = "no"
        - return empty subqueries list

        Only return structured output.
        User Query: {query}"""

    structured_llm = llm.with_structured_output(QueryAnalysis)
    result = structured_llm.invoke(decomposition_prompt)

    if result.needs_decomposition == "no":
        return {"needs_decomposition": "no", "subqueries": [query]}
    return {"needs_decomposition": "yes", "subqueries": result.subqueries}
    