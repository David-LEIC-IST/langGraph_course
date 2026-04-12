# LangGraph crash course

Hands-on notebooks for learning [LangGraph](https://langchain-ai.github.io/langgraph/)—building stateful graphs and simple agents step by step.

**Status:** This course is a work in progress. Content may be incomplete, reorganized, or updated as it grows.

## What's here

| Folder | Purpose |
|--------|---------|
| `Graphs/` | Walkthrough notebooks (hello world, multiple inputs, sequential agents, conditional agents). |
| `Exercises/` | Practice exercises to reinforce concepts learned in Graphs. |
| `Agents/` | Example agent implementations using LangGraph. |

## Getting started

1. Use Python 3.10+ (recommended for LangGraph).
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Open the project in Jupyter, VS Code, or PyCharm and run the notebooks from `Graphs/` first, then `Exercises/`.

## Key concepts covered

- State management with TypedDict
- Building and compiling graphs with StateGraph
- Handling message flows and conversation history
- Conditional routing between nodes
- Integrating LLMs (Ollama) into graph workflows

## Resources

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain documentation](https://python.langchain.com/)
