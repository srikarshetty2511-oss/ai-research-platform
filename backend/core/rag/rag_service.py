from langchain_core.messages import HumanMessage, SystemMessage
from core.llm.groq_service import GroqLLMService
from core.rag.vector_store import VectorStoreService


class RAGService:

    def __init__(self):
        self.llm_service = GroqLLMService()
        self.vector_service = VectorStoreService()
        self.vector_store = self.vector_service.load_store()

    def ask(self, question: str) -> str:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(question)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        Use the following context to answer the question.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """

        return self.llm_service.generate_response(prompt)