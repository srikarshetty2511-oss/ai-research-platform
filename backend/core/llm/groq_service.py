import os
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import LangChainException


class GroqLLMService:
    """
    Service responsible for interacting with Groq LLM.
    This class abstracts the LLM logic away from Django views.
    """

    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
        )

    def generate_response(self, user_message: str) -> str:
        """
        Generates a response from the LLM.
        """

        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        messages: List = [
            SystemMessage(
                content="You are a helpful AI assistant. Be clear and concise."
            ),
            HumanMessage(content=user_message),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content

        except LangChainException as e:
            raise RuntimeError(f"LLM processing error: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Unexpected error during LLM call: {str(e)}")