from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

def reranker(state):
    query = state["query"]
    docs = state["retrieved_docs"]
    reranked_docs = compressor.compress_documents(
        documents=docs,
        query=query
    )
    return {"reranked_docs": reranked_docs}