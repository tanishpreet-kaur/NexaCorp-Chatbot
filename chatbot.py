# import necessary libraries
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS


# load environment variables
load_dotenv()

# initialize LLM 
model = init_chat_model("google_genai:gemini-2.5-flash")

# load the required document
file_path = "NexaCorp_Enterprise_Policy_Handbook_v5.2.docx"
def load_documents(file_path: str):
    loader = UnstructuredWordDocumentLoader(
            file_path,
            mode="elements"  
        )
    return loader.load()

# split the document into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# create embeddings
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# create a vector store 
def create_vector_store(chunks, embeddings):
    import time
    
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    batch_size = 20
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        all_embeddings.extend(embeddings.embed_documents(batch))
        time.sleep(1)

    return FAISS.from_embeddings(
        list(zip(texts, all_embeddings)),
        embeddings,
        metadatas=metadatas
    )

# create retriever
def create_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
# build retriever function
def build_retriever(file_path: str):
    docs = load_documents(file_path)
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    retriever = create_retriever(vector_store)
    return retriever

retriever = build_retriever(file_path)
query = "What is the leave policy?"
docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content[:300])