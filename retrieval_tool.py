from langchain.tools import tool

def create_retrieval_tool(vector_store):
    """
    Create a retrieval tool using the vector store.

    Args:
        vector_store: The vector store object.

    Returns:
        tool: The retrieval tool.
    """
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context