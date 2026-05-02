from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def setup_embeddings(model: str = "nomic-embed-text:latest"):
    """
    Set up Ollama embeddings.

    Args:
        model (str): The model to use for embeddings.

    Returns:
        OllamaEmbeddings: The embeddings object.
    """
    return OllamaEmbeddings(model=model)

def get_vector_store(embeddings, persist_dir: str = "./chroma_db", collection_name: str = "example_collection"):
    """
    Get or create a Chroma vector store.
    
    Chroma automatically handles both creating new and loading existing stores.

    Args:
        embeddings: The embeddings function.
        persist_dir (str): Directory to persist the vector store.
        collection_name (str): Name of the collection.

    Returns:
        Chroma: The vector store object.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vector_store

def add_documents_to_store(vector_store, documents):
    """
    Add documents to the vector store.

    Args:
        vector_store: The vector store object.
        documents: List of documents to add.

    Returns:
        list: List of IDs of added documents.
    """
    ids = vector_store.add_documents(documents=documents)
    return ids