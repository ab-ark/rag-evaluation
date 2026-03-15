"""
LLM judge for AbArk RAG Eval.
Uses OpenAI-compatible API to evaluate responses.
"""

import json
import logging
import os
from typing import Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Calls an LLM to score RAG outputs.
    Compatible with OpenAI, Azure OpenAI, or any OpenAI-format endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def score(self, system_prompt: str, user_prompt: str) -> Tuple[float, str]:
        """
        Returns (score 0.0-1.0, reason string).
        Expects LLM to return JSON: {"score": float, "reason": str}
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._build_headers(),
                    json=payload,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                score = float(parsed.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                reason = parsed.get("reason", "")
                return score, reason
        except Exception as e:
            logger.error(f"LLMJudge error: {e}")
            return 0.0, f"Error: {str(e)}"
