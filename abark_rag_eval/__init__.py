"""
AbArk RAG Eval — Production-grade RAG evaluation framework.
Supports faithfulness, relevance, context precision/recall, hallucination scoring.
"""

__version__ = "0.1.0"
__author__ = "AbArk"

from .evaluators.base import BaseEvaluator
from .evaluators.rag_evaluator import RAGEvaluator
from .metrics.faithfulness import FaithfulnessMetric
from .metrics.answer_relevance import AnswerRelevanceMetric
from .metrics.context_precision import ContextPrecisionMetric
from .metrics.context_recall import ContextRecallMetric

__all__ = [
    "RAGEvaluator",
    "BaseEvaluator",
    "FaithfulnessMetric",
    "AnswerRelevanceMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
]
