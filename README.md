# LangGraph crash course

Hands-on notebooks for learning [LangGraph](https://langchain-ai.github.io/langgraph/)—building stateful graphs and simple agents step by step.

**Status:** Complete! A comprehensive course covering fundamental to advanced LangGraph concepts.

## What's here

| Folder | Purpose |
|--------|---------|
| `Graphs/` | 5 walkthrough notebooks: Hello World, Multiple Inputs, Sequential Agent, Conditional Agent, and Looping. |
| `Exercises/` | 5 practice exercises corresponding to each graph concept for hands-on learning. |
| `Agents/` | 6 complete agent implementations: Basic Bot, Memory Agent, ReAct Agent, Document Drafter, and RAG Agents (OpenAI & Llama). |
| `rag_content/` | Sample documents and ChromaDB for RAG agent demonstrations. |

## Prerequisites

- Python 3.10+ (required for LangGraph)
- Basic understanding of Python and asynchronous programming
- Familiarity with LLM concepts helpful but not required
- API keys for OpenAI (for some agents) or Groq (for free Llama access)

## Getting started

1. Clone or download this repository
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up your API keys in the `Agents/.env` file:
   - Copy `.env` and add your `OPENAI_API_KEY` and/or `GROQ_API_KEY`
   - Groq provides free access to Llama models

4. Open the project in Jupyter, VS Code, or PyCharm and run the notebooks from `Graphs/` first, then `Exercises/`.

## Course structure

### Module 1: Graph Fundamentals
- **Graph 1:** Hello World - Basic state management and graph creation
- **Exercise 1:** Practice basic graph concepts

### Module 2: Advanced Graph Patterns  
- **Graph 2:** Multiple Inputs - Handling complex state structures
- **Graph 3:** Sequential Agent - Building multi-step workflows
- **Exercise 2:** Multiple inputs practice
- **Exercise 3:** Sequential agent implementation

### Module 3: Control Flow
- **Graph 4:** Conditional Agent - Decision making in graphs
- **Graph 5:** Looping - Iteration and termination logic
- **Exercise 4:** Conditional routing challenges
- **Exercise 5:** Number guessing game with loops

### Module 4: Real-World Agents
- **Agent 1:** Basic conversational bot
- **Agent 2:** Memory-enabled agent
- **Agent 3:** ReAct (Reason+Act) agent
- **Agent 4:** Document drafter with human collaboration
- **Agent 5:** RAG agents with OpenAI and Llama

## Key concepts covered

- State management with TypedDict
- Building and compiling graphs with StateGraph
- Handling message flows and conversation history
- Conditional routing between nodes
- Looping logic and iteration control
- Tool integration and function calling
- Memory management in agents
- RAG (Retrieval-Augmented Generation) implementation
- Human-AI collaboration patterns
- Integrating multiple LLM providers (OpenAI, Groq/Llama, Ollama)

## Resources

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain documentation](https://python.langchain.com/)
- [Groq API](https://console.groq.com/) - Free Llama model access
- [Ollama](https://ollama.ai/) - Local LLM hosting
- [ChromaDB](https://www.trychroma.com/) - Vector database for RAG
