from data_pipeline.data_ingestion import bm25_retriever, parent_retriever
from langchain_classic.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, parent_retriever],
    weights=[0.4, 0.6]
)

def retrieve(state):
    queries = state.get("subqueries") or [state["query"]]
    all_docs = []
    for q in queries:
        docs = ensemble_retriever.invoke(q)
        all_docs.extend(docs)

    seen = set()
    unique_docs = []

    for doc in all_docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    return {"retrieved_docs": unique_docs}