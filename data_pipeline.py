# import necessary libraries
import re
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

# load environment variables
load_dotenv()

# file path
file_path = "NexaCorp_Enterprise_Policy_Handbook_v5.2.docx"

# load the required document
def load_document(file_path):
    try:
        loader = UnstructuredWordDocumentLoader(
            file_path,
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
            else:
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