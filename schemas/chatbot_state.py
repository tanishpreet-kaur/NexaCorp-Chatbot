from typing import TypedDict,List,Literal

class ChatbotState(TypedDict):
    query:str
    subqueries:List[str] | None
    needs_decomposition:Literal["yes","no"]
    retrieved_docs:list
    reranked_docs:list
    final_answer:str