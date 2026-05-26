from data_pipeline.data_ingestion import parent_retriever, bm25_retriever
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langfuse.decorators import observe 

# load environment variables
load_dotenv()

# initialise LLM
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

# create graph state
class ChatbotState(TypedDict):
    query: str
    subqueries: List[str] | None
    needs_decomposition: Literal["yes", "no"]
    retrieved_docs: list
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

        Rules:
        1. Answer only using the provided context.
        2. If the information is unavailable, reply exactly: "I could not find this information in the provided documents."
        3. Do not use outside knowledge or infer missing details.
        4. Start with a direct answer in 1–2 sentences.
        5. Then provide concise bullet points grouped by topic.
        6. Keep the response between 200-250 words.
        7. Mention conflicting information if present.
        8. Avoid copying large text from the context.
        9. Add citations in this format: [Section: <section number and name>, Subsection: <subsection number and name>]. 
        10. If multiple statements come from the same subsection, group them together and cite only once at the end of that group.
        11. Do not repeat identical citations across bullets.

        Context:
        {context}

        Question:
        {query}
        """

    result = llm.invoke(answer_prompt).content
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

def ask_chatbot(query: str):
    result = chatbot.invoke({"query": query})
    return result["final_answer"]
