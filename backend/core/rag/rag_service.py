from langchain_core.messages import HumanMessage, SystemMessage
from core.rag.document_loader import DocumentLoader
from core.llm.groq_service import GroqLLMService
from core.rag.vector_store import get_vector_store


class RAGService:

    def __init__(self):
        self.llm_service = GroqLLMService()
        self.vector_store = get_vector_store()

    def ask(self, question: str) -> str:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are an AI assistant describing Srikar based strictly on the provided documents.

Rules:
- Always answer in third person (use "he" or "Srikar").
- Do NOT use first person.
- Do NOT copy sentences exactly from the context.
- If numeric scores (like GPA or marks) appear in the context, interpret them qualitatively 
  (e.g., "performed exceptionally well", "graduated with distinction", 
  "maintained strong academic performance").
- Keep the tone professional and suitable for interviews or recruiter evaluation.
- If the answer is not in the context, say the information is not available.

Context:
{context}

Question:
{question}
"""

        return self.llm_service.generate_response(prompt)

