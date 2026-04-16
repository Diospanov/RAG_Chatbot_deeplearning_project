Build a complete deep learning course project: a Retrieval-Augmented Generation (RAG) chatbot with an LLM.

Project goal:
Create a production-style RAG pipeline over a custom knowledge base with:
1) document ingestion
2) chunking and preprocessing
3) embeddings + vector store
4) retrieval
5) grounded generation with citations
6) evaluation
7) simple chat UI

Important requirements:
- Use Python
- Organize the code into modules:
  - ingest/
  - retrieval/
  - generation/
  - ui/
  - eval/
- Include a README.md with setup and run instructions
- Include requirements.txt
- Use environment variables for API keys
- Do not hardcode secrets
- Make the code clean, commented, and beginner-readable

Tech stack:
- Backend language: Python 3.11+
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector store: FAISS
- UI: Streamlit
- Document formats: PDF and DOCX
- LLM provider: make it configurable
  - support OpenAI if OPENAI_API_KEY exists
  - otherwise provide a fallback stub or clearly marked placeholder function
- Evaluation:
  - retrieval precision@5
  - faithfulness / answer relevance placeholders or RAGAS integration if easy

Functional requirements:

1. Document ingestion
- Load files from a data/raw/ folder
- Support PDF and DOCX
- Extract:
  - source filename
  - title if available
  - date if available
  - full text
- Save normalized processed documents

2. Chunking
- Implement two chunking strategies:
  - fixed-size chunking with overlap
  - recursive or sentence-aware chunking
- Configurable chunk size between 100 and 512 tokens
- Configurable overlap in 10–25% range
- Preserve metadata for every chunk
- Store chunk index within each document

3. Embeddings and indexing
- Embed all chunks with sentence-transformers/all-MiniLM-L6-v2
- Build and save a FAISS index
- Store accompanying metadata in a serializable format like JSON

4. Retrieval
- For a user query, embed the query
- Retrieve top-5 most similar chunks
- Return scores, chunk text, and metadata
- Support retrieval from multiple source documents

5. Generation and grounding
- Create a system prompt that forces the model to:
  - answer only from retrieved context
  - cite source document names for factual claims
  - say exactly: "I cannot find this in the provided documents."
    if context is insufficient
- Build generation code that assembles:
  - system prompt
  - retrieved context
  - user question

6. UI
- Build a minimal Streamlit app
- Features:
  - text input for question
  - answer display
  - visible source citations
  - optional section showing retrieved chunks

7. Evaluation
- Add eval/dataset_sample.json with sample QA format
- Implement precision@5
- Add experiment logging support
- Create a simple experiment log template

8. Report support
- Add a docs/ or report_assets/ folder
- Include a Mermaid architecture diagram in a markdown file
- Add a markdown file explaining GPT-2 vs BERT in the context of RAG:
  - why BERT-like models are used for retrieval
  - why GPT-style models are used for generation

Project structure target:

rag_chatbot/
  ingest/
  retrieval/
  generation/
  ui/
  eval/
  data/
    raw/
    processed/
    index/
  docs/
  README.md
  requirements.txt
  .env.example
  main.py

Implementation expectations:
- Make the project runnable
- Use functions and classes where appropriate
- Include error handling for missing files and empty retrieval results
- Prefer simple maintainable code over fancy abstractions
- Add docstrings
- Make file paths Windows-friendly too

What I want from you:
1) First create the full folder structure
2) Then create all core files with working code
3) Then create requirements.txt
4) Then create .env.example
5) Then create README.md
6) Then summarize what was built and what I still need to configure manually

Important:
- If any part cannot be fully completed without my API key or sample documents, create placeholders and clearly explain them
- Do not leave TODO comments without also implementing a reasonable default
- Keep the project suitable for a university submission