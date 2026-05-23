from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from .models import LLMResult, Message


MODEL_META_REGISTRY: dict[str, dict[str, Any]] = {
    "gpt5.1": {
        "model": "gpt-5.1",
        "reasoning": {"effort": "medium"},
    },
    "gpt4.1": {
        "model": "gpt-4.1",
        "reasoning": None,
    },
}


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
        model_meta: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        store: bool = False,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model_meta = model_meta or os.environ.get("OPENAI_MODEL_META")
        self.reasoning: dict[str, Any] | None = None

        if self.model_meta:
            if self.model_meta not in MODEL_META_REGISTRY:
                raise ValueError(
                    f"Unknown model_meta {self.model_meta!r}. "
                    f"Available values: {sorted(MODEL_META_REGISTRY)}"
                )
            meta = MODEL_META_REGISTRY[self.model_meta]
            self.model = str(meta["model"])
            self.reasoning = meta.get("reasoning")
        else:
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

        # Explicit model overrides the model name but not the reasoning capability profile
        # unless model_meta is omitted. This allows quick experiments with exact model names.
        if model is not None:
            self.model = model
            if self.model_meta is None:
                self.reasoning = None

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.store = store

    def generate(self, messages: list[Message],     
                 *,
                tools: list[dict[str, Any]] | None = None,) -> LLMResult:
        api_input = [m.to_dict() for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": api_input,
            "store": self.store,
        }
        if self.reasoning is not None:
            kwargs["reasoning"] = self.reasoning
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if tools is not None:
            kwargs["tools"] = tools

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

    def generate(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None) -> LLMResult:
        self.counter += 1
        joined = "\n\n".join(m.content for m in messages)
        if "feature_engineering_function_code" in joined or "assembler agent" in joined:
            text = """feature_engineering_function_code: |
  def tsc_isolated_intersection_feature_vector(tls_id: str, *, cache: dict | None = None) -> list[float]:
      return [0.0]
expert_feature_mapping:
  - expert_id: expert_01
    aspect: Dry-run aspect
    feature_group_name: dry_run_feature
    feature_names:
      - dry_run_feature_0
    feature_indices:
      start: 0
      end: 1
    dimensions:
      - index: 0
        name: dry_run_feature_0
        semantic_meaning: Dry-run assembled feature.
        expected_scale: '[0, 1]'
    source_expert_output: dry-run
assembly_notes:
  assumptions:
    - Dry-run assembler output.
  unresolved_items: []
"""
        elif "implementation_status" in joined or "feature_code" in joined:
            text = """implementation_status: pseudocode_with_clear_algorithm
feature_code: |
  def dry_run_expert_feature() -> list[float]:
      return [0.0]
feature_outputs:
  - name: dry_run_feature
    semantic_meaning: Dry-run expert feature.
    output_shape: scalar
    output_scale: '[0, 1]'
required_inputs: []
dependencies:
  imports: []
  cache_keys: []
unresolved_items: []
scenario_reflection:
  uses_specific_scenario: false
  reflected_scenario_properties: []
  reason_if_generic: Dry-run output.
alignment_notes: Dry-run code synthesis response.
assembler_notes: Dry-run code piece for control-flow testing.
"""
        elif "feature_plan:" in joined or "scenario-specific feature plan" in joined:
            text = """feature_plan:
  feature_family_name: Dry-run feature family
  responsible_topic_application: Dry-run application of the expert topic.
  referenced_approaches:
    - name: null
      contribution_to_plan: No external reference used in dry run.
  novelty_or_distinction: Dry-run distinction.
  design_rationale:
    - Dry-run rationale.
  scenario_specific_observations:
    - Dry-run scenario observation.
  required_observables: []
  computation_strategy: Dry-run computation strategy.
  expected_outputs:
    - name: dry_run_feature
      meaning: Dry-run feature.
      shape: scalar
      expected_scale: '[0, 1]'
  normalization_plan: Dry-run normalization.
  implementation_notes_for_code_step: Dry-run implementation note.
"""
        elif "clusters:" in joined or "Current expert topic collection" in joined or "topic_collection" in joined:
            text = "no_overlap: true\nclusters: []\nflags: []\n"
        else:
            text = (
                f"aspect: Dry-run aspect {self.counter}\n"
                "aspect_summary: Dry-run summary generated for control-flow testing.\n"
            )
        return LLMResult(text=text, provider="dry_run", model="dry_run")
