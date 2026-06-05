from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from aprompt.llm import LLMClient, OpenAIResponsesClient
from aprompt.models import LLMResult, Message
from aprompt.phase3_orchestrator import Phase3Orchestrator


class Phase3DryRunClient(LLMClient):
    """No-cost mock client for phase-3 control-flow testing."""

    def __init__(self, expert_ids: list[str]) -> None:
        self.expert_ids = expert_ids or ["expert_01"]
        self.counter = 0

    def generate(self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None) -> LLMResult:
        self.counter += 1
        joined = "\n\n".join(m.content for m in messages)
        if "Phase-3 evaluator task" in joined:
            return LLMResult(text=self._evaluator_output(), provider="dry_run", model="phase3_dry_run")
        if "Phase-3 expert" in joined:
            return LLMResult(text=self._expert_output(), provider="dry_run", model="phase3_dry_run")
        if "feature_engineering_function_code" in joined or "assembler agent" in joined:
            return LLMResult(text=self._assembler_output(), provider="dry_run", model="phase3_dry_run")
        return LLMResult(text="ok: true\n", provider="dry_run", model="phase3_dry_run")

    def _evaluator_output(self) -> str:
        lines = [
            "phase3_evaluator_feedback:",
            "  no_issues: false",
            "  global_summary:",
            f"    expert_feature_dim: {len(self.expert_ids)}",
            "    baseline_feature_dim: 1",
            "    expert_stats_summary: |",
            "      Dry-run expert statistics summary.",
            "    baseline_stats_summary: |",
            "      Dry-run baseline statistics summary.",
            "    overall_observation: |",
            "      Dry-run routes the first expert to stats_update and leaves others unchanged.",
            "  expert_feedback:",
        ]
        for i, expert_id in enumerate(self.expert_ids):
            action = "stats_update" if i == 0 else "no_change"
            severity = "low" if i == 0 else "none"
            lines.extend(
                [
                    f'    - expert_id: "{expert_id}"',
                    f'      feature_group_name: "dry_run_group_{i}"',
                    f'      action: "{action}"',
                    f'      severity: "{severity}"',
                    "      block_index_range:",
                    f"        start: {i}",
                    f"        end: {i + 1}",
                    "      block_summary: |",
                    f"        Dry-run block summary for {expert_id}.",
                    "      relevant_feature_semantics:",
                    f'        - feature_name: "dry_run_feature_{i}"',
                    "          semantic_meaning: |",
                    "            Dry-run semantic meaning.",
                    "      routed_feature_rows:",
                ]
            )
            if action == "stats_update":
                lines.extend(
                    [
                        f"        - idx: {i}",
                        '          issue_type: "excessive_compression"',
                        '          severity: "low"',
                        "          stats:",
                        "            mean: 0.0",
                        "            std: 0.0",
                        "            min: 0.0",
                        "            max: 0.0",
                        "            p5: 0.0",
                        "            p50: 0.0",
                        "            p95: 0.0",
                        "            dead_frac: 1.0",
                        "            nan: 0",
                        "            inf: 0",
                        "            n_samples: 1",
                        "          evidence: |",
                        "            Dry-run evidence.",
                        "          why_not_automatically_bad: |",
                        "            Dry-run self-check requested.",
                    ]
                )
            else:
                lines.append("        []")
            lines.extend(
                [
                    "      baseline_reference_summary: |",
                    "        Dry-run baseline reference.",
                    "      evaluator_message_to_expert: |",
                    "        Dry-run evaluator message.",
                ]
            )
        return "\n".join(lines) + "\n"

    def _expert_output(self) -> str:
        expert_id = self.expert_ids[0]
        return f'''expert_id: "{expert_id}"
correction_decision: "stats_only_code_update"
reuse_previous_code: false
self_critique:
  evaluator_claim_summary: |
    Dry-run claim.
  refutation_attempts:
    - |
      Dry-run refutation attempt.
  final_diagnosis: |
    Dry-run applies a minimal code-only update.
  affected_dimensions:
    - idx: 0
      issue_type: "excessive_compression"
      decision: "adjust_distribution"
      reason: |
        Dry-run reason.
updated_feature_plan: null
updated_expert_feature_code:
  implementation_status: "pseudocode_with_clear_algorithm"
  feature_code: |
    def dry_run_phase3_feature() -> list[float]:
        return [0.5]
  feature_outputs:
    - name: "dry_run_phase3_feature"
      semantic_meaning: |
        Dry-run corrected feature.
      output_shape: "scalar"
      output_scale: "[0, 1]"
  required_inputs: []
  dependencies:
    imports: []
    cache_keys: []
  unresolved_items: []
  scenario_reflection:
    uses_specific_scenario: false
    reflected_scenario_properties: []
    reason_if_generic: |
      Dry-run generic implementation.
  alignment_notes: |
    Dry-run alignment note.
  assembler_notes: |
    Dry-run assembler note.
no_change_notes: null
'''

    def _assembler_output(self) -> str:
        lines = [
            "feature_engineering_function_code: |",
            "  def tsc_isolated_intersection_feature_vector(tls_id: str, *, cache: dict | None = None) -> list[float]:",
            "      return [0.5]",
            "expert_feature_mapping:",
        ]
        for i, expert_id in enumerate(self.expert_ids):
            lines.extend(
                [
                    f'  - expert_id: "{expert_id}"',
                    f'    aspect: "Dry-run aspect {i}"',
                    f'    feature_group_name: "dry_run_group_{i}"',
                    "    feature_names:",
                    f'      - "dry_run_feature_{i}"',
                    "    feature_indices:",
                    f"      start: {i}",
                    f"      end: {i + 1}",
                    "    dimensions:",
                    f"      - index: {i}",
                    f'        name: "dry_run_feature_{i}"',
                    "        semantic_meaning: |",
                    "          Dry-run assembled feature.",
                    '        expected_scale: "[0, 1]"',
                    "    source_expert_output: |",
                    "      dry-run",
                ]
            )
        lines.extend(
            [
                "assembly_notes:",
                "  assumptions:",
                "    - |",
                "      Dry-run assembler output.",
                "  unresolved_items: []",
            ]
        )
        return "\n".join(lines) + "\n"


def _expert_ids_from_phase12_run(phase12_run_dir: Path) -> list[str]:
    mapping_path = phase12_run_dir / "expert_feature_mapping.yaml"
    if not mapping_path.exists():
        return ["expert_01"]
    obj = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or []
    ids = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and isinstance(item.get("expert_id"), str):
                ids.append(item["expert_id"])
    return sorted(dict.fromkeys(ids)) or ["expert_01"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot phase-3 statistical correction prompting.")
    parser.add_argument("--phase12-run-dir", required=True)
    parser.add_argument("--expert-stats-csv", required=True)
    parser.add_argument("--baseline-stats-csv", required=True)
    parser.add_argument("--baseline-feature-description", required=True)
    parser.add_argument("--evaluator-template", required=True)
    parser.add_argument("--expert-template", required=True)
    parser.add_argument("--assembler-template", required=True)
    parser.add_argument("--task-vars", required=True, help="Task-specific variables needed to render the assembler prompt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-meta", default="gpt5.1", choices=["gpt5.1", "gpt4.1"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--assembler-web-search", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    phase12_run_dir = Path(args.phase12_run_dir)
    if args.dry_run:
        llm = Phase3DryRunClient(_expert_ids_from_phase12_run(phase12_run_dir))
    else:
        llm = OpenAIResponsesClient(
            model=args.model,
            model_meta=args.model_meta,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            store=False,
        )

    orchestrator = Phase3Orchestrator(
        phase12_run_dir=phase12_run_dir,
        expert_stats_csv=args.expert_stats_csv,
        baseline_stats_csv=args.baseline_stats_csv,
        baseline_feature_description=args.baseline_feature_description,
        evaluator_template_path=args.evaluator_template,
        expert_template_path=args.expert_template,
        assembler_template_path=args.assembler_template,
        task_variables_path=args.task_vars,
        llm_client=llm,
        output_dir=args.output_dir,
        interactive=not args.non_interactive,
        enable_assembler_web_search=args.assembler_web_search,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
