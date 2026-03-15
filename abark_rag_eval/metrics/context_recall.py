"""
Context Recall — measures how much of the ground truth answer is covered by retrieved contexts.
Requires ground_truth in the RAGSample.
"""

from . import BaseMetric
from ..models import RAGSample, MetricScore

SYSTEM_PROMPT = """You are an expert RAG evaluation judge.
Given a ground truth answer and retrieved contexts, assess what fraction of the
ground truth information is covered by the retrieved contexts.

Score Rules:
- 1.0: All key facts in the ground truth appear in the contexts.
- 0.5-0.9: Most ground truth facts are covered.
- 0.0-0.4: Key facts from ground truth are missing from the contexts.

Return ONLY valid JSON:
{"score": <float 0.0 to 1.0>, "reason": "<brief explanation>"}"""


class ContextRecallMetric(BaseMetric):
    """Measures recall of retrieved contexts against ground truth answer."""

    name = "context_recall"

    def score(self, sample: RAGSample) -> MetricScore:
        if not sample.ground_truth:
            return MetricScore(
                metric_name=self.name,
                score=0.0,
                reason="No ground_truth provided — context recall cannot be computed.",
            )

        context_block = "\n\n".join(
            f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(sample.contexts)
        )
        user_prompt = f"""Question: {sample.query}

Ground Truth Answer: {sample.ground_truth}

Retrieved Contexts:
{context_block}

What fraction of the ground truth answer is covered by these contexts?
Return JSON with score and reason."""

        score_val, reason = self.judge.score(SYSTEM_PROMPT, user_prompt)
        return MetricScore(metric_name=self.name, score=score_val, reason=reason)
