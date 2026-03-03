from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

_vector_store_instance = None


def get_vector_store():
    global _vector_store_instance

    if _vector_store_instance is None:
        service = VectorStoreService()
        _vector_store_instance = service.load_store()

    return _vector_store_instance

class VectorStoreService:

    def __init__(self, persist_directory: str = "chroma_db"):
        base_dir = os.getcwd()
        self.persist_directory = os.path.join(base_dir, "chroma_db")
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


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