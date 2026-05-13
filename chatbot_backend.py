from data_pipeline import parent_retriever, bm25_retriever, vectorstore
import json
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
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
    needs_decomposition: str
    reranked_docs: list
    final_answer: str

# reranker node
def reranker(state: ChatbotState):
    query = state["query"]
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, parent_retriever],
        weights=[0.4, 0.6]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=compressor
    )
    reranked_docs = compression_retriever.invoke(query)
    return {"reranked_docs": reranked_docs}

# query decomposition
class QueryAnalysis(BaseModel):
    needs_decomposition: Literal["yes", "no"] = Field(description="Whether the query should be decomposed into multiple subqueries")
    subqueries: List[str] = Field(description="List of generated subqueries")
    
def query_decomposition(state: ChatbotState):
    query = state["query"]
    decomposition_prompt = PromptTemplate(
        template=""" You are a query analysis assistant for a Retrieval-Augmented Generation (RAG) chatbot. 
        Your task is to determine whether the user query should be decomposed into multiple subqueries.

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

        User Query:
        {query}
        """,
        input_variables=["query"])
    
    structured_llm = llm.with_structured_output(QueryAnalysis)
    decomposition_chain = (decomposition_prompt | structured_llm)
    result = decomposition_chain.invoke({"query": query})

    if result.needs_decomposition == "no":
        return {
            "needs_decomposition": "no",
            "subqueries": [query]
        }
    return {
        "needs_decomposition": "yes",
        "subqueries": result.subqueries
    }
    
# generate answer node
def generate_answer(state: ChatbotState):
    query = state["query"]
    docs = state["reranked_docs"]
    context = "\n\n".join([doc.page_content for doc in docs])
    
    answer_prompt = PromptTemplate(
        template="""
            You are a helpful AI assistant.
            Answer ONLY using the provided context.
            Do not create content that is not supported by the context. If you don't know the answer, say you don't know.
            Context:
            {context}
            Question:
            {query}
            """,
        input_variables=["context", "query"]
    )

    chain = (answer_prompt | llm | StrOutputParser())
    result = chain.invoke({"context": context, "query": query})
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

app = graph.compile()

result = app.invoke({
    "query": "Explain company values and organisational structure"
})

print(result["final_answer"])