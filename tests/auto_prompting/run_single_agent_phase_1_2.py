from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aprompt.llm import LLMClient, OpenAIResponsesClient
from aprompt.models import LLMResult, Message
from aprompt.single_agent_phase12_orchestrator import SingleAgentPhase12Orchestrator


class SingleAgentDryRunClient(LLMClient):
    """No-cost mock client for the single-agent phase-1+2 ablation flow."""

    def __init__(self, virtual_expert_ids: list[str]) -> None:
        self.virtual_expert_ids = virtual_expert_ids
        self.counter = 0

    def generate(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None) -> LLMResult:
        self.counter += 1
        joined = "\n\n".join(m.content for m in messages)

        if "feature_engineering_function_code" in joined or "assembler agent" in joined:
            text = self._assembler_output()
        elif "expert_code_pieces:" in joined and "expert_feature_code:" in joined:
            text = self._code_pieces_output()
        elif "expert_feature_plans:" in joined and "feature_plan:" in joined:
            text = self._feature_plans_output()
        elif "updated_topics:" in joined:
            text = "updated_topics: []\n"
        elif "Current expert topic collection" in joined or "topic_collection" in joined:
            text = "no_overlap: true\nclusters: []\nflags: []\n"
        elif "topics:" in joined and "virtual topic" in joined.lower():
            text = self._topic_set_output()
        else:
            text = self._topic_set_output()

        return LLMResult(text=text, provider="dry_run", model="single_agent_dry_run")

    def _topic_set_output(self) -> str:
        lines = ["topics:"]
        for i, expert_id in enumerate(self.virtual_expert_ids, start=1):
            lines.extend(
                [
                    f'  - expert_id: "{expert_id}"',
                    f'    aspect: "Dry-run virtual aspect {i}"',
                    "    aspect_summary: |",
                    f"      Dry-run summary for virtual topic slot {expert_id}.",
                ]
            )
        return "\n".join(lines) + "\n"

    def _feature_plans_output(self) -> str:
        lines = ["expert_feature_plans:"]
        for expert_id in self.virtual_expert_ids:
            lines.extend(
                [
                    f'  - expert_id: "{expert_id}"',
                    "    feature_plan:",
                    f'      feature_family_name: "Dry-run feature family {expert_id}"',
                    "      responsible_topic_application: |",
                    f"        Dry-run scenario application for {expert_id}.",
                    "      referenced_approaches:",
                    '        - name: "none"',
                    "          contribution_to_plan: |",
                    "            No external reference used in dry run.",
                    "      novelty_or_distinction: |",
                    "        Dry-run distinction.",
                    "      design_rationale:",
                    "        - |",
                    "          Dry-run rationale.",
                    "      scenario_specific_observations:",
                    "        - |",
                    "          Dry-run scenario observation.",
                    "      required_observables: []",
                    "      computation_strategy: |",
                    "        Dry-run computation strategy.",
                    "      expected_outputs:",
                    f'        - name: "dry_run_feature_{expert_id}"',
                    "          meaning: |",
                    "            Dry-run feature.",
                    '          shape: "scalar"',
                    '          expected_scale: "[0, 1]"',
                    "      normalization_plan: |",
                    "        Dry-run normalization.",
                    "      implementation_notes_for_code_step: |",
                    "        Dry-run implementation note.",
                ]
            )
        return "\n".join(lines) + "\n"

    def _code_pieces_output(self) -> str:
        lines = ["expert_code_pieces:"]
        for expert_id in self.virtual_expert_ids:
            lines.extend(
                [
                    f'  - expert_id: "{expert_id}"',
                    "    expert_feature_code:",
                    '      implementation_status: "pseudocode_with_clear_algorithm"',
                    "      feature_code: |",
                    f"        def dry_run_feature_{expert_id}() -> list[float]:",
                    "            return [0.0]",
                    "      feature_outputs:",
                    f'        - name: "dry_run_feature_{expert_id}"',
                    "          semantic_meaning: |",
                    "            Dry-run expert feature.",
                    '          output_shape: "scalar"',
                    '          output_scale: "[0, 1]"',
                    "      required_inputs: []",
                    "      dependencies:",
                    "        imports: []",
                    "        cache_keys: []",
                    "      unresolved_items: []",
                    "      scenario_reflection:",
                    "        uses_specific_scenario: false",
                    "        reflected_scenario_properties: []",
                    "        reason_if_generic: |",
                    "          Dry-run output.",
                    "      alignment_notes: |",
                    "        Dry-run code synthesis response.",
                    "      assembler_notes: |",
                    "        Dry-run code piece for control-flow testing.",
                ]
            )
        return "\n".join(lines) + "\n"

    def _assembler_output(self) -> str:
        first_id = self.virtual_expert_ids[0] if self.virtual_expert_ids else "expert_01"
        return f'''feature_engineering_function_code: |
  def tsc_isolated_intersection_feature_vector(tls_id: str, *, cache: dict | None = None) -> list[float]:
      return [0.0]
expert_feature_mapping:
  - expert_id: "{first_id}"
    aspect: "Dry-run aspect"
    feature_group_name: "dry_run_feature"
    feature_names:
      - "dry_run_feature_0"
    feature_indices:
      start: 0
      end: 1
    dimensions:
      - index: 0
        name: "dry_run_feature_0"
        semantic_meaning: |
          Dry-run assembled feature.
        expected_scale: "[0, 1]"
    source_expert_output: |
      dry-run
assembly_notes:
  assumptions:
    - |
      Dry-run assembler output.
  unresolved_items: []
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-agent virtual-slot phase-1+2 ablation prompting.")
    parser.add_argument("--template", required=True, help="Path to prompt template YAML with single_agent templates.")
    parser.add_argument("--task-vars", required=True, help="Path to task variable YAML.")
    parser.add_argument("--run-dir", default="runs/single_agent_run_001", help="Directory for logs and final output.")
    parser.add_argument("--num-experts", type=int, default=5, help="Number of virtual topic slots.")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-refine-count", type=int, default=1)
    parser.add_argument(
        "--model-meta",
        default="gpt5.1",
        choices=["gpt5.1", "gpt4.1"],
        help="Simple model profile. gpt5.1 uses reasoning effort medium; gpt4.1 omits reasoning.",
    )
    parser.add_argument(
        "--feature-plan-web-search",
        action="store_true",
        help="Enable OpenAI web_search tool for the single-agent scenario-specific feature-plan prompt.",
    )
    parser.add_argument(
        "--assembler-web-search",
        action="store_true",
        help="Enable OpenAI web_search tool for assembler implementation/API resolution.",
    )
    parser.add_argument("--model", default=None, help="Optional exact OpenAI model override.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--non-interactive", action="store_true", help="Do not stop for manual review.")
    parser.add_argument("--dry-run", action="store_true", help="Use no-cost mock LLM responses to test control flow.")
    args = parser.parse_args()

    virtual_expert_ids = [f"expert_{i:02d}" for i in range(1, args.num_experts + 1)]
    if args.dry_run:
        llm = SingleAgentDryRunClient(virtual_expert_ids)
    else:
        llm = OpenAIResponsesClient(
            model=args.model,
            model_meta=args.model_meta,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            store=False,
        )

    orchestrator = SingleAgentPhase12Orchestrator(
        template_path=args.template,
        task_variables_path=args.task_vars,
        llm_client=llm,
        run_dir=Path(args.run_dir),
        num_experts=args.num_experts,
        max_rounds=args.max_rounds,
        interactive=not args.non_interactive,
        max_refine_count=args.max_refine_count,
        enable_feature_plan_web_search=args.feature_plan_web_search,
        enable_assembler_web_search=args.assembler_web_search,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
