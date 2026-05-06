from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .models import LLMResult, Message


class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: list[Message]) -> LLMResult:
        raise NotImplementedError


class OpenAIResponsesClient(LLMClient):
    """OpenAI Responses API adapter.

    This class is intentionally the only OpenAI-specific module in the scaffold.
    The orchestrator depends only on LLMClient.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        store: bool = False,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.store = store

    def generate(self, messages: list[Message]) -> LLMResult:
        api_input = [m.to_dict() for m in messages]
        kwargs = {
            "model": self.model,
            "input": api_input,
            "store": self.store,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens

        response = self.client.responses.create(**kwargs)
        text = getattr(response, "output_text", None)
        if text is None:
            text = str(response)
        return LLMResult(text=text, raw=response, provider="openai", model=self.model)


class DryRunClient(LLMClient):
    """A no-cost client for checking rendering and stop points.

    It returns intentionally minimal YAML. Use this only to test control flow.
    """

    def __init__(self) -> None:
        self.counter = 0

    def generate(self, messages: list[Message]) -> LLMResult:
        self.counter += 1
        joined = "\n\n".join(m.content for m in messages)
        if "clusters:" in joined or "Current expert topic collection" in joined or "topic_collection" in joined:
            text = "no_overlap: true\nclusters: []\nflags: []\n"
        else:
            text = f"aspect: Dry-run aspect {self.counter}\naspect_summary: Dry-run summary generated for control-flow testing.\n"
        return LLMResult(text=text, provider="dry_run", model="dry_run")
