"""
Answer Relevance Metric — measures how well the answer addresses the question.
"""

from . import BaseMetric
from ..models import RAGSample, MetricScore

SYSTEM_PROMPT = """You are an expert RAG evaluation judge.
Assess whether the AI-generated answer actually addresses the user's question.

Score Rules:
- 1.0: The answer directly and completely addresses the question.
- 0.5-0.9: The answer partially addresses the question.
- 0.0-0.4: The answer is off-topic or misses the point of the question.

Return ONLY valid JSON:
{"score": <float 0.0 to 1.0>, "reason": "<brief explanation>"}"""


class AnswerRelevanceMetric(BaseMetric):
    """Checks if the answer is relevant to the original question."""

    name = "answer_relevance"

    def score(self, sample: RAGSample) -> MetricScore:
        user_prompt = f"""Question: {sample.query}

Answer: {sample.answer}

Does the answer directly address the question? Return JSON with score and reason."""

        score_val, reason = self.judge.score(SYSTEM_PROMPT, user_prompt)
        return MetricScore(metric_name=self.name, score=score_val, reason=reason)
