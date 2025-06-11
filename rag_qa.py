# rag_qa.py

import config
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. 加载嵌入模型和向量库
embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_PATH)
vectorstore = Chroma(persist_directory=config.VECTOR_DB_DIR, embedding_function=embedding_model)

# 2. 加载本地 LLM
llm = GPT4All(model=config.MODEL_PATH, verbose=False)

# 3. 自定义 Prompt（可选）
prompt_template = PromptTemplate.from_template(
    "You are a helpful document assistant. Based on the following document excerpts, answer the user's question.\n\n"
    "Document content:\n{context}\n\n"
    "User's question: {question}\n"
    "Please answer concisely and accurately in English:"
)

# 4. 构建 RetrievalQA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True  # 用于调试检索结果
)

# 5. 启动问答 CLI
print("🤖 启动完成，输入你的问题，输入 exit 退出。")
while True:
    query = input("🧠 请输入问题：")
    if query.lower() in ("exit", "quit"):
        break
    result = qa_chain({"query": query})
    print(f"\n💡 回答：{result['result']}\n")

    # Debug（可选：显示被匹配的文档片段）
    print("🔍 匹配的文档片段：")
    for doc in result["source_documents"]:
        print(f"---\n{doc.page_content[:500]}\n...")
