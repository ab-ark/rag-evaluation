"""
Context Precision — measures what fraction of retrieved contexts are actually relevant to the question.
"""

from . import BaseMetric
from ..models import RAGSample, MetricScore

SYSTEM_PROMPT = """You are an expert RAG evaluation judge.
For each provided context, determine if it is relevant to answering the question.
Count the relevant contexts vs total contexts.

Return ONLY valid JSON:
{"score": <float 0.0 to 1.0>, "reason": "<brief explanation>", "relevant_count": <int>}"""


class ContextPrecisionMetric(BaseMetric):
    """Measures precision of retrieved contexts — how many are actually useful."""

    name = "context_precision"

    def score(self, sample: RAGSample) -> MetricScore:
        context_block = "\n\n".join(
            f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(sample.contexts)
        )
        user_prompt = f"""Question: {sample.query}

Retrieved Contexts:
{context_block}

How many of these contexts are relevant to answering the question?
Total contexts: {len(sample.contexts)}
Return JSON with score (relevant_count / total_count), reason, and relevant_count."""

        score_val, reason = self.judge.score(SYSTEM_PROMPT, user_prompt)
        return MetricScore(
            metric_name=self.name,
            score=score_val,
            reason=reason,
            details={"total_contexts": len(sample.contexts)},
        )
