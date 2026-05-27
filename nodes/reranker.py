from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=6)

def reranker(state):
    queries = state.get("subqueries") or [state["query"]]
    docs = state["retrieved_docs"]
    all_reranked = []
    for q in queries:
        reranked = compressor.compress_documents(
            documents=docs,
            query=q
        )
        all_reranked.extend(reranked)

    seen = set()
    unique_docs = []

    for doc in all_reranked:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    return {"reranked_docs": unique_docs}