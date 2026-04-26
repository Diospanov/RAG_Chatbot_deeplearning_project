# RAG Chatbot with LLMs

A production-ready Retrieval-Augmented Generation (RAG) pipeline that connects document ingestion, vector search, and a large language model to build a grounded, citing chatbot over a custom knowledge base.

## Architecture

```
Documents (PDF/TXT/MD/HTML/DOCX)
        │
        ▼
   [Ingestion]          ← ingest/loader.py
        │
        ▼
   [Chunking]           ← ingest/chunker.py  (fixed-size + sentence-aware)
        │
        ▼
   [Embedding]          ← retrieval/embedder.py  (sentence-transformers)
        │
        ▼
  [Vector Store]        ← retrieval/vector_store.py  (ChromaDB)
        │
        ▼
  [Retrieval]           ← retrieval/retriever.py  (top-k cosine ANN)
        │
        ▼
  [Generation]          ← generation/generator.py  (Anthropic / OpenAI / Ollama)
        │
        ▼
   [Chat UI]            ← ui/app.py  (Gradio)
```

## Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd rag_chatbot
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (if using Anthropic) | Claude API key |
| `OPENAI_API_KEY` | Yes (if using OpenAI) | OpenAI API key |
| `LLM_PROVIDER` | No | `anthropic` (default), `openai`, or `ollama` |
| `OLLAMA_MODEL` | No | Model name for Ollama (e.g. `llama3`) |
| `CHROMA_PERSIST_DIR` | No | ChromaDB storage path (default: `./chroma_db`) |
| `COLLECTION_NAME` | No | ChromaDB collection name (default: `rag_docs`) |

### 3. Ingest documents

Put your documents in `data/docs/` (supports `.pdf`, `.txt`, `.md`, `.html`, `.docx`), then run:

```bash
python -m ingest.pipeline --docs-dir data/docs --chunk-strategy sentence
```

Options:
- `--chunk-strategy`: `fixed` or `sentence` (default: `sentence`)
- `--chunk-size`: target chunk size in tokens (default: `256`)
- `--chunk-overlap`: overlap in tokens (default: `50`)

### 4. Run the Chat UI

```bash
python -m ui.app
```

Then open `http://localhost:7860` in your browser.

### 5. Run evaluation

```bash
python -m eval.evaluate --qa-file data/eval_qa.json --output eval/results.json
```

## Project Structure

```
rag_chatbot/
├── ingest/
│   ├── loader.py          # Multi-format document loader
│   ├── chunker.py         # Fixed-size & sentence-aware chunking
│   └── pipeline.py        # Orchestrates ingestion end-to-end
├── retrieval/
│   ├── embedder.py        # Sentence-transformer embeddings
│   ├── vector_store.py    # ChromaDB interface
│   └── retriever.py       # Top-k semantic retrieval
├── generation/
│   ├── generator.py       # LLM generation with grounding prompt
│   └── prompts.py         # System prompt templates
├── ui/
│   └── app.py             # Gradio chat interface
├── eval/
│   ├── evaluate.py        # RAGAS-style evaluation runner
│   ├── metrics.py         # Precision@k, faithfulness, relevance
│   └── experiment_log.md  # Ablation experiments
├── scripts/
│   └── compare_chunking.py  # Chunking strategy comparison script
├── data/
│   ├── docs/              # Put your documents here
│   └── eval_qa.json       # 30+ QA evaluation pairs (template)
├── .env.example
├── requirements.txt
└── README.md
```

## Running Experiments

To reproduce the chunking comparison:

```bash
python scripts/compare_chunking.py --docs-dir data/docs --qa-file data/eval_qa.json
```

## License

For educational use only.
