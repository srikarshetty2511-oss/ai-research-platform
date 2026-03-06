import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

load_dotenv()

# 1️⃣ Load PDF
loader = PyPDFLoader("your_pdf.pdf")
documents = loader.load()

# 2️⃣ Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_documents(documents)

# 3️⃣ Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4️⃣ Vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
)

# 5️⃣ Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6️⃣ Ask question
question = "What is the main topic?"
relevant_docs = retriever.get_relevant_documents(question)

context = "\n\n".join([doc.page_content for doc in relevant_docs])

# 7️⃣ LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

prompt = f"""
Use this context to answer the question.
If answer not found, say you don't know.

Context:
{context}

Question:
{question}
"""

response = llm.invoke([HumanMessage(content=prompt)])

print(response.content)