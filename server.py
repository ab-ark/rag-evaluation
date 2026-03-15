"""
AbArk RAG Eval — FastAPI server.
Exposes REST endpoints for evaluation runs.
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from abark_rag_eval import RAGEvaluator
from abark_rag_eval.models import RAGSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AbArk RAG Eval API",
    description="Production-grade RAG evaluation framework by AbArk",
    version="0.1.0",
)


class EvalRequest(BaseModel):
    samples: List[dict]
    metrics: Optional[List[str]] = None
    model: str = "gpt-4o-mini"


class EvalResponse(BaseModel):
    total_samples: int
    results: List[dict]
    summary: dict


@app.get("/health")
def health():
    return {"status": "ok", "service": "abark-rag-eval"}


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    try:
        samples = [
            RAGSample(
                query=s["query"],
                answer=s["answer"],
                contexts=s["contexts"],
                ground_truth=s.get("ground_truth"),
                metadata=s.get("metadata", {}),
            )
            for s in req.samples
        ]
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing field in sample: {e}")

    evaluator = RAGEvaluator(metrics=req.metrics, model=req.model)
    results = evaluator.evaluate(samples)

    metric_names = [m.name for m in evaluator.metrics]
    summary = {}
    for name in metric_names:
        scores = [s.score for r in results for s in r.scores if s.metric_name == name]
        summary[name] = round(sum(scores) / len(scores), 4) if scores else 0.0

    return EvalResponse(
        total_samples=len(results),
        results=[r.to_dict() for r in results],
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
