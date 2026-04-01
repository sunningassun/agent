from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# 1. 初始化本地 Ollama 大模型（兼容 OpenAI API）
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)

# 2. 加载三国演义文本
loader = TextLoader("sanguoyanyi.txt", encoding='utf-8')
docs = loader.load()

# 3. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(docs)

# 4. 初始化向量嵌入模型
embedding = HuggingFaceEmbeddings(model_name='models/AI-ModelScope/bge-large-zh-v1___5')

# 5. 构建 FAISS 向量库 + 检索器
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever()

# 6. 构建提示词模板
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_message = SystemMessagePromptTemplate.from_template(
    "根据以下已知信息回答用户问题。\n 已知信息{context}"
)
human_message = HumanMessagePromptTemplate.from_template(
    "用户问题：{question}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message,
])

# 7. 构建检索问答链
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": chat_prompt}
)

# 8. 用户问题
user_question = "五虎上将有哪些？"

# 9. 查看检索到的相关文档
print("=" * 50)
print("检索到的相关文档：")
print("=" * 50)
related_docs = retriever.invoke(user_question)
for i, doc in enumerate(related_docs):
    print(f"\n文档 {i+1}:")
    print(doc.page_content)
    print("-" * 40)

# 10. 模型回答
print("\n" + "=" * 50)
print("模型回答：")
print("=" * 50)
result = qa.invoke(user_question)
print(result['result'])  # 只输出干净的答案