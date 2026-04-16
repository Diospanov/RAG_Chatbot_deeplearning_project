# RAG Chatbot Course Project

A beginner-friendly Retrieval-Augmented Generation (RAG) chatbot project in Python. The project ingests PDF and DOCX files, chunks and embeds them with `sentence-transformers/all-MiniLM-L6-v2` when the model is available locally or can be downloaded, indexes them with FAISS when available, retrieves top-5 chunks for a query, generates grounded answers with citations, and provides a Streamlit chat UI.

## Project Structure

```text
.
|-- main.py
|-- README.md
|-- requirements.txt
|-- .env.example
|-- rag_chatbot/
|   |-- ingest/
|   |-- retrieval/
|   |-- generation/
|   |-- ui/
|   |-- eval/
|   |-- data/
|   |   |-- raw/
|   |   |-- processed/
|   |   `-- index/
|   `-- docs/
`-- spec.md
```

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API key if you want OpenAI-backed generation:

```powershell
copy .env.example .env
```

4. The repository already includes two small starter DOCX files in `rag_chatbot/data/raw/` for smoke testing. Replace or add to them with your own PDF and DOCX files for the real project.

## Run The Pipeline

### 1. Ingest documents

```powershell
python main.py ingest
```

This creates `rag_chatbot/data/processed/documents.json`.

### 2. Build chunks and FAISS index

```powershell
python main.py index --strategy sentence --chunk-size 256 --overlap 0.15
```

This creates:

- `rag_chatbot/data/processed/chunks_sentence.json`
- `rag_chatbot/data/index/sentence_index.faiss`
- `rag_chatbot/data/index/sentence_metadata.json`

On platforms where a FAISS wheel is not available through pip, the project automatically falls back to a NumPy-based vector index while keeping the same file locations and retrieval flow.
If the embedding model cannot be downloaded, the project falls back to a deterministic local hashing embedder so the pipeline remains runnable offline.

### 3. Ask a question from the command line

```powershell
python main.py ask "What does the report say about evaluation?"
```

If the processed corpus or index is missing, the CLI will try to build the missing artifacts automatically.

### 4. Launch the Streamlit UI

```powershell
streamlit run rag_chatbot/ui/app.py
```

### 5. Run retrieval evaluation

```powershell
python main.py evaluate --dataset rag_chatbot/eval/dataset_sample.json
```

Evaluation appends experiment summaries to `rag_chatbot/eval/experiments/experiment_log.jsonl`.

## Environment Variables

- `OPENAI_API_KEY`: optional, enables OpenAI grounded generation
- `OPENAI_MODEL`: optional, defaults to `gpt-4o-mini`

If `OPENAI_API_KEY` is not set, the project uses a clearly marked extractive fallback generator so the rest of the RAG pipeline still works.

## Notes About Evaluation

The included `dataset_sample.json` is a template. Update the questions and `expected_sources` values so they match the files you actually place in `rag_chatbot/data/raw/`.

Implemented metrics:

- precision@5
- placeholder faithfulness record
- placeholder answer relevance record

## Supporting Docs

- Architecture diagram: [rag_chatbot/docs/architecture.md](rag_chatbot/docs/architecture.md)
- GPT vs BERT in RAG: [rag_chatbot/docs/gpt2_vs_bert_in_rag.md](rag_chatbot/docs/gpt2_vs_bert_in_rag.md)

## Common Issues

- `No supported files found`: add PDF or DOCX files to `rag_chatbot/data/raw/`
- `sentence-transformers is required`: run `pip install -r requirements.txt`
- embedding model download fails: the app will fall back to a local hashing embedder, but retrieval quality will be lower than `all-MiniLM-L6-v2`
- `FAISS index was not found`: run `python main.py index`
- `I cannot find this in the provided documents.`: retrieval returned insufficient evidence for the question
