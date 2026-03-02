# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage
# import os

# class RAGAgent:

#     def __init__(self):
#         self.llm = ChatGroq(
#             model="llama-3.1-8b-instant",
#             groq_api_key=os.getenv("GROQ_API_KEY"),
#             streaming=True,
#         )

#     async def stream_response(self, user_input):
#         async for chunk in self.llm.astream(
#             [HumanMessage(content=user_input)]
#         ):
#             if chunk.content:
#                 yield chunk.content