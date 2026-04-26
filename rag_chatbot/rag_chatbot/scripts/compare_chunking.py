"""
scripts/compare_chunking.py
----------------------------
Compare fixed-size vs sentence-aware chunking by:
  1. Ingesting the same documents with both strategies
  2. Running retrieval on eval questions
  3. Reporting precision@5 for each strategy

Run:
    python scripts/compare_chunking.py --docs-dir data/docs --qa-file data/eval_qa.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compare(docs_dir: str, qa_file: str, top_k: int = 5) -> None:
    from ingest.loader import load_directory
    from ingest.chunker import chunk_documents
    from retrieval.embedder import embed_query, embed_texts
    from eval.metrics import precision_at_k, recall_at_k

    documents = load_directory(docs_dir)
    if not documents:
        logger.error("No documents found in %s", docs_dir)
        return

    with open(qa_file) as f:
        qa_pairs = json.load(f)

    strategies = ["fixed", "sentence"]
    report = {}

    for strategy in strategies:
        logger.info("\n=== Strategy: %s ===", strategy)

        chunks = chunk_documents(documents, strategy=strategy, chunk_size=256, chunk_overlap=50)
        logger.info("Produced %d chunks", len(chunks))

        # Build a tiny in-memory index (numpy cosine search)
        texts = [c["text"] for c in chunks]
        logger.info("Embedding %d chunks…", len(texts))
        embeddings = embed_texts(texts, show_progress=True)

        import numpy as np
        emb_matrix = np.array(embeddings)   # (N, D)

        precisions, recalls = [], []

        for qa in qa_pairs:
            q_emb = np.array(embed_query(qa["question"]))
            sims = emb_matrix @ q_emb                   # cosine (normalized)
            top_idx = np.argsort(sims)[::-1][:top_k]
            retrieved = [
                {"text": chunks[i]["text"], "score": float(sims[i])}
                for i in top_idx
            ]
            gt_passages = qa.get("ground_truth_passages", [qa.get("ground_truth_answer", "")])
            p = precision_at_k(retrieved, gt_passages[0] if gt_passages else "", k=top_k)
            r = recall_at_k(retrieved, gt_passages, k=top_k)
            precisions.append(p)
            recalls.append(r)

        mean_p = sum(precisions) / len(precisions)
        mean_r = sum(recalls) / len(recalls)
        report[strategy] = {"precision_at_5": round(mean_p, 4), "recall_at_5": round(mean_r, 4)}
        logger.info("Precision@5: %.4f  |  Recall@5: %.4f", mean_p, mean_r)

    print("\n" + "=" * 50)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 50)
    for strategy, metrics in report.items():
        print(f"\n  Strategy: {strategy}")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
    print("=" * 50)

    # Save
    out = Path("eval/chunking_comparison.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs-dir", default="data/docs")
    p.add_argument("--qa-file", default="data/eval_qa.json")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    compare(args.docs_dir, args.qa_file, top_k=args.top_k)
