from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_MODEL = "gpt-4.1-2025-04-14"

# USD per 1M tokens. Used only for rough usage reporting.
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4.1-2025-04-14": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini-2025-04-14": {"input": 0.4, "output": 1.6},
}


@dataclass
class UsageTotals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
        }


class OpenAIChatClient:
    """Minimal wrapper around OpenAI Chat Completions with usage accounting."""

    def __init__(self, api_key: Optional[str], model: str = DEFAULT_MODEL) -> None:
        resolved_api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not resolved_api_key:
            raise RuntimeError(
                "OpenAI API key is required. Set OPENAI_API_KEY or pass --api-key."
            )

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency 'openai'. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=resolved_api_key)
        self.model = model
        self.usage_totals = UsageTotals()

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.0,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self._client.chat.completions.create(**kwargs)
        self._record_usage(response.usage)
        return response

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)

        pricing = PRICING.get(self.model, PRICING.get(DEFAULT_MODEL, {"input": 0.0, "output": 0.0}))
        est_cost = (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

        self.usage_totals.calls += 1
        self.usage_totals.input_tokens += input_tokens
        self.usage_totals.output_tokens += output_tokens
        self.usage_totals.total_tokens += total_tokens
        self.usage_totals.cost += est_cost
