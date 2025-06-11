# semantic_search.py

import config
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 加载嵌入模型和向量库
embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_PATH)
vectorstore = Chroma(persist_directory=config.VECTOR_DB_DIR, embedding_function=embedding_model)

# 2. 启动语义检索 CLI
print("🔍 启动完成（不含 LLM）。输入你的问题，直接查看匹配片段，输入 exit 退出。")
while True:
    query = input("🧠 请输入问题：")
    if query.lower() in ("exit", "quit"):
        break

    # 3. 进行向量相似度查询（取前 3 个相似片段）
    results = vectorstore.similarity_search(query, k=3)

    print("\n📚 匹配到的文档片段：")
    for i, doc in enumerate(results, 1):
        print(f"\n🔹 片段 {i}：\n{doc.page_content[:500]}\n...")
