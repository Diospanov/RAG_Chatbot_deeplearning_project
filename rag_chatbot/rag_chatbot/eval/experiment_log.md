# Experiment Log

All experiments follow the format:
**component changed | param before | param after | metric before | metric after | observation**

Baseline: sentence chunking, chunk_size=256, overlap=50, top_k=5, embedding=all-MiniLM-L6-v2

---

## Experiment 1 — Chunk size

| Field | Value |
|---|---|
| Component | Chunking: chunk_size |
| Param before | 128 tokens |
| Param after | 256 tokens |
| Metric (Precision@5) before | 0.52 |
| Metric (Precision@5) after | 0.68 |
| Observation | Larger chunks contain more context per passage, improving retrieval relevance. Too small chunks lose sentence context and split related facts across chunks. |

---

## Experiment 2 — Chunking strategy

| Field | Value |
|---|---|
| Component | Chunking strategy |
| Param before | fixed |
| Param after | sentence-aware |
| Metric (Precision@5) before | 0.61 |
| Metric (Precision@5) after | 0.68 |
| Observation | Sentence-aware chunking preserves semantic units. Fixed chunking often splits sentences mid-thought, reducing embedding quality. Sentence strategy is consistently better at the same token budget. |

---

## Experiment 3 — Chunk overlap

| Field | Value |
|---|---|
| Component | Chunking: chunk_overlap |
| Param before | 0 tokens (no overlap) |
| Param after | 50 tokens |
| Metric (Recall@5) before | 0.54 |
| Metric (Recall@5) after | 0.71 |
| Observation | Overlap prevents information at chunk boundaries from being "lost." Without overlap, facts spanning two chunks are partially retrievable; with overlap, the entire span appears in at least one chunk. |

---

## Experiment 4 — Retrieval top-k

| Field | Value |
|---|---|
| Component | Retrieval: top_k |
| Param before | 3 |
| Param after | 5 |
| Metric (Recall@k) before | 0.60 |
| Metric (Recall@k) after | 0.71 |
| Observation | Increasing k from 3→5 improved recall significantly with minimal increase in context length. Going beyond 8 showed diminishing returns and risk of context dilution (lower signal-to-noise in the LLM prompt). |

---

## Experiment 5 — Embedding model

| Field | Value |
|---|---|
| Component | Embedding model |
| Param before | all-MiniLM-L6-v2 (384d) |
| Param after | all-mpnet-base-v2 (768d) |
| Metric (Answer Relevance) before | 0.72 |
| Metric (Answer Relevance) after | 0.78 |
| Observation | The larger mpnet model produces higher-quality embeddings and improves answer relevance, at the cost of ~4x slower encoding. For datasets >50k chunks, the latency tradeoff may not be worth it. MiniLM is the recommended default. |

---

## Summary

The final configuration uses:
- **Strategy**: sentence-aware
- **chunk_size**: 256 tokens
- **overlap**: 50 tokens (~20%)
- **top_k**: 5
- **embedding**: all-MiniLM-L6-v2
