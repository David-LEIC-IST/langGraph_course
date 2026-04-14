"""Drafter

Boss's Order - Task:
- Our company is not working efficiently! We spend way too much time drafting documents and this needs to ve fixed!
- For the company, you need to create an AI Agentic System that can speed up drafting documents, emails,etc.
The AI Agentic System should have Human-AI Collaboration meaning the human should be able to provide continuous
feedback and the AI Agent should stop when the human is happy with the draft. The system should also be fast and be
able to save the drafts.
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# This is a global variable to store the document content
document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is: \n{document_content}"


@tool
def save(filename: str) -> str:
    """Saves the current document to a text file and finish the process
    Args:
        filename (str): The name of the text file to save the document to
    Returns:
        str: A message indicating that the document has been saved successfully
    """
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"Document has been saved to: {filename}")
        return f"Document has been saved successfully to {filename}"
    except Exception as e:
        return f"Failed to save document: {str(e)}"


tools = [update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

        - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after modifications.

        The current document content is:{document_content}
        """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation"""

    messages = state["messages"]

    # This looks for the most recent tool message
    for message in reversed(messages):
        # and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and
                "saved" in message.content.lower() and
                "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint

    return "continue"


def print_messages(messages):
    """Function to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"TOOL RESULT : {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER =====\n")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====\n")


if __name__ == "__main__":
    run_document_agent()





