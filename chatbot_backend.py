from data_pipeline import parent_retriever, bm25_retriever, vectorstore
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# load environment variables
load_dotenv()

# initialise LLM
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

# create graph state
class ChatbotState(TypedDict):
    query: str
    subqueries: List[str] | None
    needs_decomposition: Literal["yes", "no"]
    reranked_docs: list
    final_answer: str

# initialize retrievers and compressor
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=4)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, parent_retriever],
    weights=[0.4, 0.6]
)

# reranker node
def reranker(state: ChatbotState):
    queries = state.get("subqueries") or [state["query"]]
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=compressor
    )
    
    all_docs = []
    for q in queries:
        docs = compression_retriever.invoke(q)
        all_docs.extend(docs)
        
    seen = set()
    unique_docs = [
        d for d in all_docs
        if not (d.page_content in seen or seen.add(d.page_content))
    ]
    return {"reranked_docs": unique_docs}

# query decomposition
class QueryAnalysis(BaseModel):
    needs_decomposition: Literal["yes", "no"] = Field(description="Whether the query should be decomposed into multiple subqueries")
    subqueries: List[str] = Field(description="List of generated subqueries")

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
    
# generate answer node
def generate_answer(state: ChatbotState):
    query = state["query"]
    docs = state["reranked_docs"]
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    answer_prompt = f"""
        You are NexaCorp's HR assistant.

        Strict rules:
        1. Answer ONLY using information present in the provided context.
        2. Do NOT use prior knowledge or make assumptions.
        3. If the answer cannot be found in the context, respond exactly: "I could not find this information in the provided documents."
        4. Do NOT infer missing details from related information.
        5. Do NOT combine multiple policies unless the context explicitly connects them.
        6. Give a direct answer in 1–2 sentences first.
        7. Then summarize important points using bullet points.
        8. Keep the response under 200-250 words unless detailed explanation is requested.
        9. Avoid copying large portions of the source text.
        10. Include only information relevant to the question.
        11. If retrieved documents contain conflicting information, mention the conflict instead of choosing one.

        Context:
        {context}

        Question:
        {query}
        """

    result = llm.invoke(answer_prompt).content

    return {"final_answer": result}

# initialize state graph
graph = StateGraph(ChatbotState)

# create nodes
graph.add_node("reranker", reranker)
graph.add_node("query_decomposition", query_decomposition)
graph.add_node("answer_generation", generate_answer)

# create edges
graph.add_edge(START, "query_decomposition")
graph.add_edge("query_decomposition", "reranker")
graph.add_edge("reranker", "answer_generation")
graph.add_edge("answer_generation", END)

chatbot = graph.compile()

while True:
    query = input("\nQuestion: ")
    if query.lower() == "exit":
        break
    result = chatbot.invoke({"query": query})
    print("\nAnswer:")
    print(result["final_answer"])
