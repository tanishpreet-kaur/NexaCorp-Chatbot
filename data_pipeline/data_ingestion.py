# import necessary libraries
import re
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

# load environment variables
load_dotenv()

# file path
BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "NexaCorp_Enterprise_Policy_Handbook_v5.2.docx"

# load the required document
def load_document(file_path):
    try:
        loader = UnstructuredWordDocumentLoader(
            str(file_path),
            mode="elements"                                 
        )
        docs = loader.load()
        return docs
    except Exception as e:
        raise RuntimeError(f"Failed to load document: {file_path}") from e
    
    
# use regular expressions to remove header/footer noise
def is_header_footer(text):
    NOISE_PATTERNS = [
        r"Version\s+\d+[\.\d]*\s*\|",
        r"Internal Restricted",
        r"Not for External Distribution",
        r"©\s*20\d{2}",
        r"CONFIDENTIAL",
        r"Policy.*Handbook"
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in NOISE_PATTERNS)


# uses regular expression to check if line belongs to TOC or not
def is_toc_line(text):
    text = text.strip()
    return bool(re.search(
        r"^(Part\s+[IVXLC]+|[\d]+\.[\d]+)\s+.*\s+\d+$",
        text,
        re.IGNORECASE
    ))
    
    
# clean the documents by removing unwanted data
def clean_documents(docs):
    cleaned_docs = []
    skip_toc = False
    removed = 0
    
    for doc in docs:
        text = doc.page_content.strip()
        lower = text.lower()
        category = doc.metadata.get("category", "")
        text = re.sub(r"\s+", " ", text)
        
        if category in ["Header", "Footer"]:
            removed += 1
            continue

        if is_header_footer(text):
            removed += 1
            continue

        if "table of contents" in lower:
            skip_toc = True
            continue

        if skip_toc:
            if re.match(r"^1\.\s", text): 
                skip_toc = False
            elif is_toc_line(text):
                continue

        if category == "Table":
            doc.metadata["type"] = "table"
        else:
            doc.metadata["type"] = "text"

        doc.page_content = text
        cleaned_docs.append(doc)
        
    return cleaned_docs


# create a structured format for cleaned documents
def build_structured_docs(docs_clean):
    structured_docs = []
    current_part = ""
    current_section = ""
    current_subsection = ""
    buffer = []

    def flush_buffer(doc_type="text"):
        if buffer:
            structured_docs.append({
                "part": current_part,
                "section": current_section,
                "subsection": current_subsection,
                "content": " ".join(buffer).strip(),
                "type": doc_type
            })
            buffer.clear()

    for doc in docs_clean:
        text = re.sub(r"\s+", " ", doc.page_content).strip()
        category = doc.metadata.get("category", "")
        if not text:
            continue

        # HEADING DETECTION
        if category == "Title":
            flush_buffer()

            # LEVEL 3
            if re.match(r"^\d+\.\d+\.\d+\s+", text):
                current_subsection = text

            # LEVEL 2
            elif re.match(r"^\d+\.\d+\s+", text):
                current_section = text
                current_subsection = ""

            # LEVEL 1
            elif re.match(r"^\d+\.\s+", text):
                current_part = text
                current_section = ""
                current_subsection = ""
            continue

        # TABLE HANDLING
        if (
            doc.metadata.get("type") == "table"
            or category == "Table"
        ):
            flush_buffer()
            structured_docs.append({
                "part": current_part,
                "section": current_section,
                "subsection": current_subsection,
                "content": text,
                "type": "table"
            })
            continue
        buffer.append(text)
    flush_buffer()
    return structured_docs


# create parent documents for hierarchical retrieval
def create_parent_docs(structured_docs):
    parent_docs = []
    for item in structured_docs:           
        full_content = f"""
            Part: {item.get("part", "")}
            Section: {item.get("section", "")}
            Subsection: {item.get("subsection", "")}
            {item["content"]}
            """.strip()
        
        doc_id = hashlib.md5(full_content.encode()).hexdigest()

        parent_docs.append(
            Document(
                page_content=full_content,
                metadata={
                    "doc_id": doc_id,
                    "source": "NexaCorp Policy Handbook v5.2",
                    "part": item.get("part", ""),
                    "section": item.get("section", ""),
                    "subsection": item.get("subsection", ""),
                    "raw_content": item["content"],
                    "type": item.get("type", "text")
                }
            )
        )
    return parent_docs


# create BM25 retriever
def create_bm25_retriever(parent_docs):
    bm25_retriever = BM25Retriever.from_documents(parent_docs)
    bm25_retriever.k = 10
    return bm25_retriever

# initialize embedding model
def load_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},   
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )
    return embeddings


# create pinecone vector store
def create_vectorstore(embeddings):
    pc = Pinecone()
    index_name = "nexacorp-rag-chatbot"
    namespace = "hr-policy-v2"
    existing_indexes = [i.name for i in pc.list_indexes()]

    # create index if not exists
    if index_name not in existing_indexes:
        print("Creating Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )
    return vectorstore


# create parent document retriever
def create_parent_retriever(vectorstore, parent_docs):
    fs = LocalFileStore("./parent_doc_store")
    store = create_kv_docstore(fs)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 10},
        child_metadata_fields=[
            "doc_id",
            "source",
            "part",
            "section",
            "subsection",
            "type"
        ]
    )

    ADD_PARENT_DOCS = False
    if ADD_PARENT_DOCS:
        print("Adding parent documents...")
        parent_retriever.add_documents(parent_docs)
        print("Parent documents added.")
    else:
        print("Using existing parent documents.")

    return parent_retriever


# FINAL RAG PIPELINE
def build_rag_pipeline():
    docs = load_document(file_path)
    cleaned_docs = clean_documents(docs)
    structured_docs = build_structured_docs(cleaned_docs)
    parent_docs = create_parent_docs(structured_docs)
    embeddings = load_embedding_model()
    vectorstore = create_vectorstore(embeddings)
    parent_retriever = create_parent_retriever(vectorstore, parent_docs)
    bm25_retriever = create_bm25_retriever(parent_docs)

    return {
        "vectorstore": vectorstore,
        "parent_retriever": parent_retriever,
        "bm25_retriever": bm25_retriever,
        "parent_docs": parent_docs
    }
    

pipeline = build_rag_pipeline()
parent_retriever = pipeline["parent_retriever"]
bm25_retriever = pipeline["bm25_retriever"]