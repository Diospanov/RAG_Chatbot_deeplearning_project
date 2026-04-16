# GPT-Style Models vs BERT-Like Models in RAG

## Why BERT-like models are used for retrieval

BERT-style encoders are designed to produce dense semantic representations of text. In a RAG pipeline, that makes them a strong fit for retrieval because we want both the user query and document chunks embedded into the same vector space. Similar meanings should land close together, even when the wording is different.

Models such as `sentence-transformers/all-MiniLM-L6-v2` are efficient, practical for course projects, and widely used for semantic search. They are encoder-style models, so they are optimized for representing text rather than continuing it word by word.

## Why GPT-style models are used for generation

GPT-style models are autoregressive generators. Their strength is producing fluent, coherent answers conditioned on a prompt. In a RAG system, we give the model the retrieved chunks as context and ask it to synthesize a grounded answer.

This division of labor works well:

- BERT-like encoders retrieve the most relevant evidence.
- GPT-style generators turn that evidence into a readable answer.

## Why both are needed in RAG

If we only use a generator, the model may hallucinate or rely on stale pretraining knowledge. If we only use a retriever, we get relevant passages but not a final answer. RAG combines the two:

- retrieval narrows the evidence set
- generation explains the answer in natural language
- citations help users trace claims back to sources

## In this project

- Retrieval uses `all-MiniLM-L6-v2` with FAISS.
- Generation uses OpenAI when `OPENAI_API_KEY` is available.
- If no API key is configured, the project falls back to a simple extractive stub so the rest of the pipeline still works.
