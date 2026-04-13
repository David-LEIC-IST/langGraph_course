"""ReAct Agent

Objectives:
1. Learn how to create Tools in LangGraph
2. How to create a ReAct Agent
3. Work with different type of Messages such as ToolMessages
4. Test out robustness of our graph

Main Goal: Create a robust ReAct Agent
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 integers together"""
    return a + b


@tool
def subtract(a: int, b: int):
    """This is a subtract function that subtracts 2 integers"""
    return a - b


@tool
def multiply(a: int, b: int):
    """This is a multiply function that multiplies 2 integers"""
    return a * b


tools = [add, subtract, multiply]

model = ChatOllama(model="llama3.2", temperature=0.3).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant,please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_messages = messages[-1]
    if not last_messages.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 40 + 12, then multiply that value by 6. Also tell me a joke later")]}

print_stream(app.stream(inputs, stream_mode="values"))