"""
Tests for AbArk RAG Eval.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch

from abark_rag_eval.models import RAGSample, EvalResult, MetricScore
from abark_rag_eval.evaluators.rag_evaluator import RAGEvaluator
from abark_rag_eval.llm_judge import LLMJudge


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_faithful():
    return RAGSample(
        query="What is the capital of France?",
        answer="The capital of France is Paris.",
        contexts=["France is a country in Western Europe. Its capital city is Paris."],
        ground_truth="Paris is the capital of France.",
    )


@pytest.fixture
def sample_hallucinated():
    return RAGSample(
        query="What is the capital of France?",
        answer="The capital of France is Berlin.",
        contexts=["France is a country in Western Europe. Its capital city is Paris."],
    )


@pytest.fixture
def mock_judge():
    judge = MagicMock(spec=LLMJudge)
    judge.score.return_value = (0.9, "Mocked: answer is grounded in context.")
    return judge


# ── Unit Tests ─────────────────────────────────────────────────────────────────

class TestRAGSample:
    def test_sample_creation(self, sample_faithful):
        assert sample_faithful.query == "What is the capital of France?"
        assert len(sample_faithful.contexts) == 1
        assert sample_faithful.ground_truth is not None

    def test_sample_no_ground_truth(self, sample_hallucinated):
        assert sample_hallucinated.ground_truth is None


class TestMetricScore:
    def test_score_bounds(self):
        score = MetricScore(metric_name="faithfulness", score=0.85, reason="test")
        assert 0.0 <= score.score <= 1.0


class TestEvalResult:
    def test_aggregate_score_empty(self, sample_faithful):
        result = EvalResult(sample=sample_faithful)
        assert result.aggregate_score == 0.0

    def test_aggregate_score_computed(self, sample_faithful):
        result = EvalResult(sample=sample_faithful, scores=[
            MetricScore("faithfulness", 0.8),
            MetricScore("answer_relevance", 0.9),
        ])
        assert abs(result.aggregate_score - 0.85) < 0.001

    def test_to_dict(self, sample_faithful):
        result = EvalResult(
            sample=sample_faithful,
            scores=[MetricScore("faithfulness", 0.9, "test reason")],
        )
        d = result.to_dict()
        assert "query" in d
        assert "scores" in d
        assert "faithfulness" in d["scores"]


class TestRAGEvaluator:
    def test_evaluator_init_default_metrics(self):
        evaluator = RAGEvaluator.__new__(RAGEvaluator)
        evaluator.metrics = []
        evaluator.max_workers = 4
        assert evaluator.max_workers == 4

    @patch("abark_rag_eval.evaluators.rag_evaluator.LLMJudge")
    def test_evaluate_returns_results(self, MockJudge, sample_faithful):
        mock_judge_instance = MagicMock()
        mock_judge_instance.score.return_value = (0.85, "good")
        MockJudge.return_value = mock_judge_instance

        evaluator = RAGEvaluator(metrics=["faithfulness"], model="gpt-4o-mini", api_key="test")
        results = evaluator.evaluate([sample_faithful])

        assert len(results) == 1
        assert results[0].sample.query == sample_faithful.query

    def test_print_summary_empty(self, capsys):
        evaluator = RAGEvaluator.__new__(RAGEvaluator)
        evaluator.metrics = []
        evaluator.print_summary([])
        captured = capsys.readouterr()
        assert "No results" in captured.out

    def test_export_json(self, tmp_path, sample_faithful):
        import json
        result = EvalResult(
            sample=sample_faithful,
            scores=[MetricScore("faithfulness", 0.9, "test")],
        )
        evaluator = RAGEvaluator.__new__(RAGEvaluator)
        evaluator.metrics = []
        out_path = str(tmp_path / "results.json")
        evaluator.export_json([result], out_path)

        with open(out_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["query"] == sample_faithful.query

    def test_export_csv(self, tmp_path, sample_faithful):
        import csv
        result = EvalResult(
            sample=sample_faithful,
            scores=[MetricScore("faithfulness", 0.9, "test")],
        )
        mock_metric = MagicMock()
        mock_metric.name = "faithfulness"

        evaluator = RAGEvaluator.__new__(RAGEvaluator)
        evaluator.metrics = [mock_metric]
        out_path = str(tmp_path / "results.csv")
        evaluator.export_csv([result], out_path)

        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["query"] == sample_faithful.query


# ── Integration-style test (skipped in CI without API key) ────────────────────

@pytest.mark.skipif(
    not __import__("os").environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_full_evaluation_integration():
    sample = RAGSample(
        query="What is machine learning?",
        answer="Machine learning is a subset of AI where algorithms learn from data.",
        contexts=["Machine learning (ML) is a branch of artificial intelligence that enables systems to learn from data."],
        ground_truth="Machine learning is an AI discipline where models learn patterns from data.",
    )
    evaluator = RAGEvaluator(metrics=["faithfulness", "answer_relevance"])
    results = evaluator.evaluate([sample])
    assert len(results) == 1
    assert results[0].aggregate_score > 0.0
