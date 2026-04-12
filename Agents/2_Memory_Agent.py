"""ChatBot

Objectives:
1. Use different message types - HumanMessage and AIMessage
2. Maintain a full conversation history using both message types
3. Use Llama3.2 model using LangChain's OllamaLLM
4. Create a sophisticated conversation loop

Main Goal: Create a form of memory for our Agent
"""

from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = OllamaLLM(model="llama3.2")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response))
    print(f"\nAI: {response}")

    return state


graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")