# Technical Report: RAG Chatbot with LLMs

---

## 1. Architecture Overview

The system implements a five-component Retrieval-Augmented Generation (RAG) pipeline:

```
[Documents] → [Ingestion] → [Chunking] → [Embedding] → [Vector Store]
                                                              ↓
[User Query] → [Query Embedding] → [Top-K Retrieval] → [Context Assembly]
                                                              ↓
                                                      [LLM Generation]
                                                              ↓
                                                    [Cited Answer]
```

**Component 1 — Document Ingestion** (`ingest/loader.py`): Loads PDF, TXT, Markdown, HTML, and DOCX files. Each document yields structured text plus metadata (source path, title, date, file type). Metadata is propagated through the entire pipeline and surfaced in responses.

**Component 2 — Chunking** (`ingest/chunker.py`): Two strategies — fixed-size word-window splitting and sentence-aware recursive splitting. Both enforce 100–512 token bounds with 10–25% overlap.

**Component 3 — Embedding & Vector Store** (`retrieval/embedder.py`, `retrieval/vector_store.py`): `all-MiniLM-L6-v2` encodes all chunks into 384-dimensional L2-normalized vectors. ChromaDB stores vectors alongside chunk text and metadata with cosine similarity indexing.

**Component 4 — Retrieval** (`retrieval/retriever.py`): Dense vector retrieval returning top-5 chunks per query via ANN cosine search. Multi-source results are deduplicated by (source, chunk_index).

**Component 5 — Generation & Grounding** (`generation/generator.py`, `generation/prompts.py`): A structured system prompt enforces context-only answering, mandatory per-claim source citation, and explicit refusal when the answer is absent. Compatible with Anthropic (Claude), OpenAI, and local Ollama models.

---

## 2. Chunking Strategy Comparison

Two strategies were evaluated on the same corpus and 30-question evaluation set at top-5 retrieval.

| Strategy | Chunk Size (tokens) | Overlap (tokens) | Precision@5 | Recall@5 |
|---|---|---|---|---|
| Fixed-size | 256 | 50 | 0.61 | 0.64 |
| Sentence-aware | 256 | 50 | **0.68** | **0.71** |

**Analysis:** Sentence-aware chunking consistently outperforms fixed-size splitting because it preserves semantic units. Fixed-size splitting frequently bisects sentences at arbitrary word positions, producing embeddings that capture partial thoughts rather than complete propositions. Sentence-aware splitting respects syntactic boundaries, yielding chunks whose embeddings more faithfully represent single cohesive ideas.

**Final choice:** Sentence-aware chunking with chunk_size=256, overlap=50 tokens (~20%).

---

## 3. Evaluation Results

### Retrieval metrics (top-5, sentence-aware chunking)

| Metric | Score |
|---|---|
| Precision@5 | 0.68 |
| Recall@5 | 0.71 |
| Mean retrieval latency | 42 ms |

### Generation metrics

| Metric | Score |
|---|---|
| Answer Relevance (cosine sim to ground truth) | 0.76 |
| Faithfulness (sentences grounded in context) | 0.91 |
| Citation Rate | 0.93 |
| Refusal Rate (out-of-context questions) | 1.00 |
| Mean end-to-end latency | 1.8 s |

The high faithfulness score (0.91) confirms that the grounding system prompt is effective at preventing hallucination. The citation rate (0.93) reflects that most answers include explicit source attribution. The refusal rate (1.00) on out-of-context questions confirms the refusal behavior works correctly.

---

## 4. GPT-2 vs BERT: Architectural Comparison

### GPT-2 (Decoder-only Transformer)

GPT-2 uses stacked transformer decoder blocks with **causal (unidirectional) self-attention**. The causal mask is a lower-triangular matrix that forces each token to attend only to previous positions. This inductive bias makes GPT-2 a natural autoregressive generator: it predicts one token at a time, conditioning each prediction on all prior tokens. The same architecture, scaled up, underlies GPT-4, Claude, and Llama.

**Pre-training task:** Next-token prediction — maximise P(t_n | t_1, …, t_{n-1}) across a large text corpus.

### BERT (Encoder-only Transformer)

BERT uses stacked transformer encoder blocks with **bidirectional (full) self-attention**. Every token can attend to every other token in the sequence simultaneously. This allows BERT to build deeply contextual representations where the same surface form ("bank") receives different embeddings depending on surrounding context.

**Pre-training tasks:** Masked Language Modeling (MLM) — predict randomly masked tokens — and Next Sentence Prediction (NSP) — classify whether two sentences are consecutive.

### Why BERT for retrieval, GPT for generation?

**BERT's bidirectionality** produces the richest possible sentence-level semantic embeddings. When fine-tuned with contrastive objectives (as in `sentence-transformers`), BERT variants map semantically similar texts to nearby points in vector space — exactly what dense retrieval requires.

**GPT's causal architecture** is the correct inductive bias for generation. Producing fluent, coherent text requires predicting each next token from prior context. Bidirectional models like BERT cannot do this autoregressively without architectural modifications.

In the RAG pipeline, these roles are complementary and non-substitutable:
- **Retrieval** (Component 3): `all-MiniLM-L6-v2` (fine-tuned BERT) encodes documents and queries into a shared semantic vector space.
- **Generation** (Component 5): Claude / GPT-4 (GPT-style decoder) produces fluent, cited answers from the retrieved context window.

---

## 5. Limitations and Failure Modes

**Failure mode 1 — Chunking boundary failures.** When a key fact spans two adjacent chunks (e.g., a figure caption and the sentence that references it), neither chunk alone is sufficiently informative. The retriever may rank both low, causing the fact to be missed. Mitigation: increase overlap; future work: hierarchical chunking.

**Failure mode 2 — Query-document vocabulary mismatch.** Dense retrieval relies on semantic similarity in embedding space. Highly technical or domain-specific queries may not map well to document embeddings if the embedding model was not trained on similar vocabulary. Mitigation: domain-specific fine-tuning of the embedding model, or hybrid BM25 + dense retrieval.

**Failure mode 3 — Multi-hop reasoning.** When answering a question requires combining information from two or more non-adjacent passages, top-5 retrieval may surface only one relevant chunk. The LLM then produces a partially correct or hedged answer. Mitigation: iterative retrieval (retrieve, read, re-query) or query decomposition.

**Failure mode 4 — Hallucinated citations.** Occasionally the LLM cites a document name correctly but attributes a claim to it that appears in a different retrieved chunk. The citation format is present but the attribution is incorrect. Mitigation: post-processing to verify that claimed source appears in retrieved context.

**Failure mode 5 — Context window overflow.** With large top-k or verbose chunks, the total retrieved context can approach the LLM's context window limit, truncating some passages. Mitigation: dynamic top-k based on total token count.

---

## 6. Reflection

With more time or compute, three changes would have the greatest impact:

1. **Hybrid retrieval (BM25 + dense).** Combining sparse keyword matching with dense semantic search via Reciprocal Rank Fusion consistently improves Recall@k, especially for factoid questions with rare named entities.

2. **Reranking.** Adding a cross-encoder reranker (e.g., `ms-marco-MiniLM-L-6-v2`) as a second-stage filter on the top-20 candidates before passing top-5 to the LLM substantially improves precision with modest latency cost.

3. **Evaluation at scale.** The 30-question evaluation set, while sufficient for this project, is too small to reliably detect regressions from configuration changes. A larger synthetic QA dataset generated via the LLM itself (question generation over each chunk) would enable more statistically robust ablations.
