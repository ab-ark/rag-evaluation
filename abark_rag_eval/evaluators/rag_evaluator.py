"""
RAGEvaluator — orchestrates all configured metrics over a dataset.
Supports parallel evaluation, CSV/JSON export, and summary reporting.
"""

import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from .base import BaseEvaluator
from ..llm_judge import LLMJudge
from ..metrics.faithfulness import FaithfulnessMetric
from ..metrics.answer_relevance import AnswerRelevanceMetric
from ..metrics.context_precision import ContextPrecisionMetric
from ..metrics.context_recall import ContextRecallMetric
from ..models import RAGSample, EvalResult

logger = logging.getLogger(__name__)


class RAGEvaluator(BaseEvaluator):
    """
    Full RAG evaluation pipeline.

    Usage:
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(samples)
        evaluator.export_csv(results, "results.csv")
        evaluator.print_summary(results)
    """

    DEFAULT_METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_workers: int = 4,
    ):
        self.judge = LLMJudge(
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        metric_names = metrics or self.DEFAULT_METRICS
        self.metrics = []
        metric_map = {
            "faithfulness": FaithfulnessMetric,
            "answer_relevance": AnswerRelevanceMetric,
            "context_precision": ContextPrecisionMetric,
            "context_recall": ContextRecallMetric,
        }
        for name in metric_names:
            if name in metric_map:
                self.metrics.append(metric_map[name](self.judge))
            else:
                logger.warning(f"Unknown metric '{name}' — skipping.")
        self.max_workers = max_workers

    def _evaluate_single(self, sample: RAGSample) -> EvalResult:
        result = EvalResult(sample=sample)
        for metric in self.metrics:
            try:
                score = metric.score(sample)
                result.scores.append(score)
                logger.debug(f"[{metric.name}] query='{sample.query[:40]}...' score={score.score:.3f}")
            except Exception as e:
                logger.error(f"Metric '{metric.name}' failed: {e}")
        return result

    def evaluate(self, samples: List[RAGSample]) -> List[EvalResult]:
        """Evaluate a list of RAG samples. Returns EvalResult for each."""
        logger.info(f"Evaluating {len(samples)} samples with {len(self.metrics)} metrics...")
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._evaluate_single, s): s for s in samples}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Sample evaluation failed: {e}")

        logger.info(f"Evaluation complete. {len(results)} results.")
        return results

    def print_summary(self, results: List[EvalResult]) -> None:
        """Print a human-readable summary table."""
        if not results:
            print("No results to summarize.")
            return

        metric_names = [m.name for m in self.metrics]
        averages = {}
        for name in metric_names:
            scores = [
                s.score
                for r in results
                for s in r.scores
                if s.metric_name == name
            ]
            averages[name] = sum(scores) / len(scores) if scores else 0.0

        print("\n" + "=" * 60)
        print("  AbArk RAG Eval — Summary Report")
        print("=" * 60)
        print(f"  Samples evaluated: {len(results)}")
        print("-" * 60)
        for name, avg in averages.items():
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            print(f"  {name:<25} {bar}  {avg:.3f}")
        overall = sum(averages.values()) / len(averages) if averages else 0.0
        print("-" * 60)
        print(f"  {'Overall Score':<25}{'':20}  {overall:.3f}")
        print("=" * 60 + "\n")

    def export_csv(self, results: List[EvalResult], path: str) -> None:
        """Export results to CSV."""
        metric_names = [m.name for m in self.metrics]
        fieldnames = ["query", "answer", "ground_truth", "aggregate_score"] + metric_names

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "query": r.sample.query,
                    "answer": r.sample.answer,
                    "ground_truth": r.sample.ground_truth or "",
                    "aggregate_score": round(r.aggregate_score, 4),
                }
                for s in r.scores:
                    row[s.metric_name] = round(s.score, 4)
                writer.writerow(row)
        logger.info(f"Results exported to {path}")

    def export_json(self, results: List[EvalResult], path: str) -> None:
        """Export results to JSON."""
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info(f"Results exported to {path}")
