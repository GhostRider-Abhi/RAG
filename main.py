from data_loader import load_and_chunk_pdf
from vector_store_setup import setup_embeddings, get_vector_store, add_documents_to_store
import os

def main():
    # Configuration
    pdf_path = "Think-And-Grow-Rich_2011-06.pdf"
    embedding_model = "nomic-embed-text:latest"
    persist_dir = "./chroma_db"
    collection_name = "example_collection"

    # Setup embeddings
    embeddings = setup_embeddings(embedding_model)

    # Check if vector store already exists
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Vector store already exists. Loading existing store...")
        vector_store = get_vector_store(embeddings, persist_dir, collection_name)
    else:
        print("Creating new vector store...")
        all_splits = load_and_chunk_pdf(pdf_path)
        vector_store = get_vector_store(embeddings, persist_dir, collection_name)
        add_documents_to_store(vector_store, all_splits)
        print("Vector store created and documents added.")

    print("Setup complete. You can now run the UI with: streamlit run ui.py")

if __name__ == "__main__":
    main()
