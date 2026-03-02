from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreService:

    def __init__(self, persist_directory: str = "chroma_db"):
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = persist_directory

    def create_store(self, documents):
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory,
        )
        vector_store.persist()
        return vector_store

    def load_store(self):
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding,
        )