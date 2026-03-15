"""
Core data models for AbArk RAG Eval.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class RAGSample:
    """A single RAG evaluation sample."""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricScore:
    """Score from a single metric."""
    metric_name: str
    score: float
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Full evaluation result for one RAG sample."""
    sample: RAGSample
    scores: List[MetricScore] = field(default_factory=list)

    @property
    def aggregate_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.sample.query,
            "answer": self.sample.answer,
            "contexts_count": len(self.sample.contexts),
            "ground_truth": self.sample.ground_truth,
            "aggregate_score": round(self.aggregate_score, 4),
            "scores": {s.metric_name: {"score": round(s.score, 4), "reason": s.reason} for s in self.scores},
            "metadata": self.sample.metadata,
        }
