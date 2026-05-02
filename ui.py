import streamlit as st
from vector_store_setup import setup_embeddings, get_vector_store
from retrieval_tool import create_retrieval_tool
from agent import setup_llm, create_rag_agent, run_agent_query

# Configuration
embedding_model = "nomic-embed-text:latest"
llm_model = "llama3.1:latest"
persist_dir = "./chroma_db"
collection_name = "example_collection"

# Setup components
embeddings = setup_embeddings(embedding_model)
vector_store = get_vector_store(embeddings, persist_dir, collection_name)
retrieve_context = create_retrieval_tool(vector_store)
tools = [retrieve_context]
llm = setup_llm(llm_model)
agent = create_rag_agent(llm, tools)

# Streamlit UI
st.title("RAG Chatbot")
st.write("Ask questions about the book!")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            response = run_agent_query(agent, query)
        st.write("**Response:**")
        st.write(response)
    else:
        st.warning("Please enter a question.")