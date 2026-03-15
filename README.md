# abark-rag-eval

> **Production-grade RAG Evaluation Framework by AbArk**
> Evaluate retrieval-augmented generation pipelines with LLM-as-judge metrics, parallel scoring, REST API, and full CI/CD.

[![CI](https://github.com/AbArk/abark-rag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/AbArk/abark-rag-eval/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- **4 core metrics** — Faithfulness, Answer Relevance, Context Precision, Context Recall
- **LLM-as-judge** — OpenAI-compatible (GPT-4o-mini, local models, Azure OpenAI)
- **Parallel evaluation** — thread-pool executor for fast batch runs
- **Multiple connectors** — feed samples from pgvector, Pinecone, Chroma, or raw JSON
- **REST API** — FastAPI server with `/evaluate` endpoint
- **Export** — CSV and JSON report generation
- **Docker ready** — single `docker-compose up` deployment

---

## Quick Start

```bash
# Install
pip install -e .

# Set your LLM judge key
export OPENAI_API_KEY=sk-...

# Run evaluation in Python
python -c "
from abark_rag_eval import RAGEvaluator
from abark_rag_eval.models import RAGSample

sample = RAGSample(
    query='What is LangChain?',
    answer='LangChain is a framework for building LLM applications.',
    contexts=['LangChain is an open-source framework that helps developers build applications powered by large language models.'],
    ground_truth='LangChain is a framework for developing LLM-powered applications.',
)

evaluator = RAGEvaluator()
results = evaluator.evaluate([sample])
evaluator.print_summary(results)
evaluator.export_csv(results, 'results.csv')
"
```

---

## REST API

```bash
# Start server
uvicorn server:app --reload

# POST /evaluate
curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "samples": [{
      "query": "What is RAG?",
      "answer": "RAG stands for Retrieval Augmented Generation.",
      "contexts": ["RAG is a technique that combines retrieval with LLM generation."],
      "ground_truth": "RAG = Retrieval Augmented Generation."
    }],
    "metrics": ["faithfulness", "answer_relevance"]
  }'
```

---

## Metrics

| Metric | Description | Requires `ground_truth` |
|---|---|---|
| `faithfulness` | Is the answer grounded in contexts? | No |
| `answer_relevance` | Does the answer address the question? | No |
| `context_precision` | What fraction of contexts are relevant? | No |
| `context_recall` | Are ground truth facts covered by contexts? | Yes |

---

## Docker

```bash
docker build -t abark-rag-eval .
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 abark-rag-eval
```

Or with Docker Compose:

```bash
docker-compose up
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Architecture

```
abark_rag_eval/
├── models.py          # RAGSample, MetricScore, EvalResult
├── llm_judge.py       # OpenAI-compatible LLM judge
├── metrics/
│   ├── faithfulness.py
│   ├── answer_relevance.py
│   ├── context_precision.py
│   └── context_recall.py
└── evaluators/
    ├── base.py
    └── rag_evaluator.py   # Main orchestrator
server.py                  # FastAPI REST server
```

---

## References & Inspiration

- [RAGAS](https://github.com/explodinggradients/ragas) — RAG evaluation metrics
- [DeepEval](https://github.com/confident-ai/deepeval) — LLM evaluation framework
- [vectara/open-rag-eval](https://github.com/vectara/open-rag-eval) — UMBRELA & AutoNuggetizer

---

## License

MIT © [AbArk](https://github.com/AbArk)
