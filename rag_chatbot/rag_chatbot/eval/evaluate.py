"""
eval/evaluate.py
----------------
Run evaluation over a QA dataset and produce a results report.

QA dataset format (JSON):
    [
        {
            "question": "What is RAG?",
            "ground_truth_answer": "RAG stands for ...",
            "ground_truth_passages": ["RAG is a technique that ..."]
        },
        ...
    ]

Run:
    python -m eval.evaluate --qa-file data/eval_qa.json --output eval/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(
    qa_path: str | Path,
    top_k: int = 5,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Evaluate the RAG pipeline on a QA dataset.
    Returns a summary dict with aggregate metrics and per-question results.
    """
    from generation.generator import RAGGenerator
    from eval.metrics import (
        precision_at_k,
        recall_at_k,
        answer_relevance,
        faithfulness,
        has_citation,
    )

    qa_path = Path(qa_path)
    with open(qa_path) as f:
        qa_pairs = json.load(f)

    logger.info("Evaluating on %d QA pairs", len(qa_pairs))
    gen = RAGGenerator(top_k=top_k)

    results = []
    agg = {
        "precision_at_k": [],
        "recall_at_k": [],
        "answer_relevance": [],
        "faithfulness": [],
        "citation_rate": [],
        "latency_s": [],
    }

    for i, qa in enumerate(qa_pairs, start=1):
        question = qa["question"]
        gt_answer = qa.get("ground_truth_answer", "")
        gt_passages = qa.get("ground_truth_passages", [gt_answer] if gt_answer else [])

        logger.info("[%d/%d] %s", i, len(qa_pairs), question[:80])
        t0 = time.time()
        result = gen.answer(question)
        latency = time.time() - t0

        p_at_k = precision_at_k(
            result["sources"],
            gt_passages[0] if gt_passages else "",
            k=top_k,
        )
        r_at_k = recall_at_k(result["sources"], gt_passages, k=top_k)
        ar = answer_relevance(result["answer"], gt_answer) if gt_answer else None
        faith = faithfulness(result["answer"], result["context"])
        citation = has_citation(result["answer"])

        row = {
            "question": question,
            "answer": result["answer"],
            "ground_truth_answer": gt_answer,
            "precision_at_k": round(p_at_k, 4),
            "recall_at_k": round(r_at_k, 4),
            "answer_relevance": round(ar, 4) if ar is not None else None,
            "faithfulness": round(faith, 4),
            "has_citation": citation,
            "is_refusal": result["is_refusal"],
            "latency_s": round(latency, 3),
            "sources": [
                {
                    "title": s["metadata"].get("title", ""),
                    "score": round(s["score"], 4),
                }
                for s in result["sources"]
            ],
        }
        results.append(row)

        agg["precision_at_k"].append(p_at_k)
        agg["recall_at_k"].append(r_at_k)
        if ar is not None:
            agg["answer_relevance"].append(ar)
        agg["faithfulness"].append(faith)
        agg["citation_rate"].append(float(citation))
        agg["latency_s"].append(latency)

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    summary = {
        "num_questions": len(qa_pairs),
        "top_k": top_k,
        "mean_precision_at_k": _mean(agg["precision_at_k"]),
        "mean_recall_at_k": _mean(agg["recall_at_k"]),
        "mean_answer_relevance": _mean(agg["answer_relevance"]),
        "mean_faithfulness": _mean(agg["faithfulness"]),
        "citation_rate": _mean(agg["citation_rate"]),
        "mean_latency_s": _mean(agg["latency_s"]),
        "results": results,
    }

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if k != "results":
            print(f"  {k:30s}: {v}")
    print("=" * 60 + "\n")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    p.add_argument("--qa-file", required=True, help="Path to QA JSON file")
    p.add_argument("--top-k", type=int, default=5, help="Retrieval top-k")
    p.add_argument("--output", default="eval/results.json", help="Output JSON path")
    args = p.parse_args()

    run_evaluation(args.qa_file, top_k=args.top_k, output_path=args.output)
