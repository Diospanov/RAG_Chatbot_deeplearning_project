# RAG Chatbot Architecture

```mermaid
flowchart TD
    A[data/raw<br/>PDF / DOCX] --> B[Ingestion Module]
    B --> C[Normalized documents.json]
    C --> D[Chunking Module<br/>fixed or sentence-aware]
    D --> E[Embedding Model<br/>all-MiniLM-L6-v2]
    E --> F[FAISS Index]
    D --> G[Chunk Metadata JSON]
    H[User Question] --> I[Query Embedding]
    I --> F
    F --> J[Top-5 Retrieved Chunks]
    G --> J
    J --> K[Grounded Prompt Builder]
    H --> K
    K --> L[LLM or Fallback Generator]
    L --> M[Answer with Citations]
    J --> N[Evaluation Module]
    M --> N
    M --> O[Streamlit UI]
    J --> O
```

## Notes

- The ingestion step extracts text and metadata from PDF and DOCX files.
- Chunk metadata is preserved across chunking, indexing, retrieval, generation, and evaluation.
- The generator is configured to refuse unsupported questions with the exact required sentence.
