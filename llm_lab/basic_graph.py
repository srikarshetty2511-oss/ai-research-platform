import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

def llm_node(state):
    question = state["question"]
    response = llm.invoke(question)
    return {"answer": response.content}

graph = StateGraph(dict)

graph.add_node("llm", llm_node)
graph.set_entry_point("llm")
graph.add_edge("llm", END)

app = graph.compile()

result = app.invoke({"question": "What is LangGraph?"})
print(result["answer"])