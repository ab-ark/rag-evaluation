"""
Faithfulness Metric — measures if the answer is grounded in the provided contexts.
Score of 1.0 = fully grounded. Score of 0.0 = hallucinated.
"""

from . import BaseMetric
from ..models import RAGSample, MetricScore

SYSTEM_PROMPT = """You are an expert RAG evaluation judge.
Your task is to assess whether an AI-generated answer is fully supported by the provided context passages.

Score Rules:
- 1.0: Every claim in the answer is directly supported by the contexts.
- 0.5-0.9: Most claims are supported, but some details are inferred or slightly unsupported.
- 0.0-0.4: Many claims in the answer are not found in the contexts (hallucinated).

Return ONLY valid JSON in this format:
{"score": <float 0.0 to 1.0>, "reason": "<brief explanation>"}"""


class FaithfulnessMetric(BaseMetric):
    """Checks if the answer only contains information from the retrieved contexts."""

    name = "faithfulness"

    def score(self, sample: RAGSample) -> MetricScore:
        context_block = "\n\n".join(
            f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(sample.contexts)
        )
        user_prompt = f"""Question: {sample.query}

Contexts:
{context_block}

Answer: {sample.answer}

Is the answer faithful to the contexts? Return JSON with score and reason."""

        score_val, reason = self.judge.score(SYSTEM_PROMPT, user_prompt)
        return MetricScore(
            metric_name=self.name,
            score=score_val,
            reason=reason,
            details={"contexts_used": len(sample.contexts)},
        )
