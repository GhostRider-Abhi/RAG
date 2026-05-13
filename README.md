# RAG PDF Q&A Prototype

A small Retrieval-Augmented Generation (RAG) proof-of-concept built with:

- `LangChain` for orchestration
- `Ollama` for embeddings and chat
- `Chroma` for persistent vector storage
- `Streamlit` for a simple web UI

## Project Structure

- `agent.py` - creates and runs the LangChain chat agent
- `data_loader.py` - loads a PDF and chunks it into retrievable documents
- `vector_store_setup.py` - configures embeddings and the Chroma vector store
- `retrieval_tool.py` - defines a retrieval tool for similarity search
- `ui.py` - Streamlit app for asking questions
- `main.py` - build/setup script for vector store initialization
- `main.ipynb` - notebook walkthrough of the RAG pipeline

## Getting Started

1. Create a Python environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your PDF file in the project root and update `main.py` if needed.

4. Build the vector store (if it does not already exist):

```bash
python main.py
```

5. Run the UI:

```bash
streamlit run ui.py
```

## Notes

- The project currently persists Chroma data under `./chroma_db`.
- The vector store should be excluded from version control.
- If the model startup is slow, cache initialization and/or choose a smaller model.

