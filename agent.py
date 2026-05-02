from langchain_ollama import ChatOllama
from langchain.agents import create_agent

def setup_llm(model: str = "llama3.1:latest", temperature: float = 0):
    """
    Set up the ChatOllama LLM.

    Args:
        model (str): The model to use.
        temperature (float): Temperature for generation.

    Returns:
        ChatOllama: The LLM object.
    """
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        validate_model_on_init=True
    )
    return llm

def create_rag_agent(llm, tools, system_prompt: str = "You have access to a tool that retrieves context from a book. Use the tool to help answer user queries."):
    """
    Create a RAG agent.

    Args:
        llm: The language model.
        tools: List of tools.
        system_prompt (str): System prompt for the agent.

    Returns:
        agent: The created agent.
    """
    agent = create_agent(llm, tools, system_prompt=system_prompt)
    return agent

def run_agent_query(agent, query: str):
    """
    Run a query through the agent and return the response.

    Args:
        agent: The agent.
        query (str): The user query.

    Returns:
        str: The agent's response.
    """
    response = ""
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        if event["messages"]:
            message = event["messages"][-1]
            if hasattr(message, 'content'):
                response = message.content
    return response