"""Simple Bot

Objectives:
1. Define state structure with a list of HumanMessage objects
2. Initialize an Ollama model using LangChain's OllamaLLM
3. Sending and handling different types of messages
4. Building and compiling the graph of the Agent

Main Goal: How to integrate LLMs in our graphs
"""

from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = OllamaLLM(model="llama3.2")


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response}")
    # Append AI response to conversation history
    state["messages"].append(HumanMessage(content=response))
    return state


graph = StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    # Add user message to conversation history
    conversation_history.append(HumanMessage(content=user_input))
    
    # Invoke agent with full conversation history
    result = agent.invoke({"messages": conversation_history})
    
    # Update conversation history with AI response
    conversation_history = result["messages"]
    
    user_input = input("Enter: ")
