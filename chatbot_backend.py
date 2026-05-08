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
from langgraph.graph import StateGraph, END

# load environment variables
load_dotenv()

# initialise LLM
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

# create graph state
class ChatbotState(TypedDict):
    query: str
    subqueries: List[str] | None
    retrieved_docs: list
    reranked_docs: list
    final_answer: str
    
# hybrid retriever node
def hybrid_retriever(state: ChatbotState):
    query = state["query"]
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, parent_retriever],
        weights=[0.4, 0.6]
    )
    docs = hybrid_retriever.invoke(query)
    return {"retrieved_docs": docs} 

# reranker node
def reranker(state: ChatbotState):
    query = state["query"]
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=reranker_model)
    
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
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
    
    result = []
    return result

# initialize state graph
graph = StateGraph(ChatbotState)

# create nodes
graph.add_node("hybrid_retriever", hybrid_retriever)
graph.add_node("reranker", reranker)
graph.add_node("query_decomposition", query_decomposition)
graph.add_node("answer generation", generate_answer)

# create edges