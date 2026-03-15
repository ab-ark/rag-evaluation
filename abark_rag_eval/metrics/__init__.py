"""Base class for all RAG evaluation metrics."""

from abc import ABC, abstractmethod
from ..models import RAGSample, MetricScore
from ..llm_judge import LLMJudge


class BaseMetric(ABC):
    """All metrics must implement score()."""

    name: str = "base_metric"

    def __init__(self, judge: LLMJudge):
        self.judge = judge

    @abstractmethod
    def score(self, sample: RAGSample) -> MetricScore:
        """Compute metric score for a single RAG sample."""
        ...
