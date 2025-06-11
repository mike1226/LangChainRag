# embeddings_loader.py

import config
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def _build_store_from_docs(docs):
    """Build and persist vector store from loaded documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_PATH)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=config.VECTOR_DB_DIR,
    )
    vectorstore.persist()
    print("✅ 向量库已构建完成并保存。")


def build_vector_store():
    """Load text file specified by ``config.DOC_PATH`` and build vector store."""
    loader = TextLoader(config.DOC_PATH)
    docs = loader.load()
    _build_store_from_docs(docs)


def build_vector_store_from_pdf(pdf_path: str):
    """Load a PDF file and store its contents in the vector store."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    _build_store_from_docs(docs)

if __name__ == "__main__":
    if config.DOC_PATH.lower().endswith(".pdf"):
        build_vector_store_from_pdf(config.DOC_PATH)
    else:
        build_vector_store()
