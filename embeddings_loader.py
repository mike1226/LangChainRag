# embeddings_loader.py

import config
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_store():
    # 1. 加载 TXT 文档
    loader = TextLoader(config.DOC_PATH)
    docs = loader.load()

    # 2. 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # 3. 构建向量库（使用本地 embedding 模型）
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_PATH)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=config.VECTOR_DB_DIR
    )
    # 4. 持久化向量库
    vectorstore.persist()
    print("✅ 向量库已构建完成并保存。")

if __name__ == "__main__":
    build_vector_store()
