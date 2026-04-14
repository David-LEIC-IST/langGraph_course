import os

from langchain_classic import text_splitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Our Embedding Model has to also be compatible with the llm
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

pdf_path = "../rag_content/Stock_Market_Performance_2024.pdf"

# Safety measure for debugging purposes
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"The PDF file {pdf_path} does not exist.")

pdf_loader = PyPDFLoader(pdf_path)

# Check if the pdf is there
try:
    pages = pdf_loader.load()
    print(f"Successfully loaded {len(pages)} pages from the PDF.")
except Exception as e:
    print(f"Failed to load PDF: {str(e)}")
    raise

# Chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"C:\Users\didar\PycharmProject\LangGraph_Course\rag_content"
collection_name = "stock_market"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Here we actually create the chroma database using our embeddings model
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    print(f"Created ChromaDB vector store")
except Exception as e:
    print(f"Failed to create vector store: {str(e)}")
    raise

# Now we create our retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5  # K is the amount of chunks to return
    }
)


@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the Stock Market Performance 2024 document"""
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document"

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Result {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls"""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools}


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response"""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t["name"]} with query: {t["args"].get("query", "No query provided")}")

        if not t["name"] in tools_dict:
            print(f"\nTool: {t["name"]} does not exist.")
            result = "Incorrect tool name, please retry and select tool from list of available tools"
        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)

graph.add_edge("retriever_agent", "llm")

graph.set_entry_node("llm")

rag_agent = graph.compile()


def running_agent():
    """Function to run the agent"""
    print("\n===== RAG AGENT =====\n")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n===== ANSWER =====")
        print(result["messages"][-1].content)


running_agent()