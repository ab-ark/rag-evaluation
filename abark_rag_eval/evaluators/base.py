"""Base evaluator for AbArk RAG Eval."""

from abc import ABC, abstractmethod
from typing import List
from ..models import RAGSample, EvalResult


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, samples: List[RAGSample]) -> List[EvalResult]:
        ...
