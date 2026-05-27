from langgraph.graph import StateGraph, END
from schemas.chatbot_state import ChatbotState
from nodes.query_decomposition import query_decomposition
from nodes.retrieval import retrieve
from nodes.reranker import reranker
from nodes.answer_generation import generate_answer

graph = StateGraph(ChatbotState)

# create nodes
graph.add_node("retrieve", retrieve)
graph.add_node("reranker", reranker)
graph.add_node("query_decomposition", query_decomposition)
graph.add_node("generate_answer", generate_answer)

# create edges
graph.set_entry_point("query_decomposition")
graph.add_edge("query_decomposition", "retrieve")
graph.add_edge("retrieve", "reranker")
graph.add_edge("reranker", "generate_answer")
graph.add_edge("generate_answer", END)

chatbot = graph.compile()