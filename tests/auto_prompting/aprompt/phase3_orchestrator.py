from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from .io_utils import append_jsonl, dump_yaml, load_yaml, write_text, write_yaml
from .llm import LLMClient
from .models import LLMCallLog, Message
from .parsing import parse_yaml_response, validate_assembler_result, validate_expert_feature_code
from .template_renderer import TemplateStore


VALID_EVALUATOR_ACTIONS = {"no_change", "stats_update", "feature_plan_review"}
VALID_EXPERT_DECISIONS = {"no_change", "stats_only_code_update", "feature_plan_and_code_update"}


class Phase3Orchestrator:
    """One-shot phase-3 statistical correction orchestrator.

    Inputs are a completed phase-1+2 run directory, expert/baseline feature
    statistics, baseline-feature semantics, phase-3 evaluator/expert templates,
    the phase-1+2 assembler template, and the task-specific variable file needed
    to render the assembler prompt.

    The assembler is intentionally called in a fresh session with the same input
    contract as phase 2. It only sees the merged current source-of-truth artifacts.
    """

    def __init__(
        self,
        *,
        phase12_run_dir: str | Path,
        expert_stats_csv: str | Path,
        baseline_stats_csv: str | Path,
        baseline_feature_description: str | Path,
        evaluator_template_path: str | Path,
        expert_template_path: str | Path,
        assembler_template_path: str | Path,
        task_variables_path: str | Path,
        llm_client: LLMClient,
        output_dir: str | Path,
        interactive: bool = True,
        enable_assembler_web_search: bool = False,
    ) -> None:
        self.phase12_run_dir = Path(phase12_run_dir)
        self.expert_stats_csv = Path(expert_stats_csv)
        self.baseline_stats_csv = Path(baseline_stats_csv)
        self.baseline_feature_description = Path(baseline_feature_description)
        self.evaluator_template_path = Path(evaluator_template_path)
        self.expert_template_path = Path(expert_template_path)
        self.assembler_template_path = Path(assembler_template_path)
        self.task_variables_path = Path(task_variables_path)
        self.llm = llm_client
        self.output_dir = Path(output_dir)
        self.interactive = interactive
        self.enable_assembler_web_search = enable_assembler_web_search

        self.evaluator_templates = TemplateStore(load_yaml(self.evaluator_template_path))
        self.expert_templates = TemplateStore(load_yaml(self.expert_template_path))
        self.assembler_templates = TemplateStore(load_yaml(self.assembler_template_path))

        task_yaml = load_yaml(self.task_variables_path)
        self.task_variables = dict(task_yaml.get("variables", {}))
        self.task_id = task_yaml.get("task_id", "task")

        self.call_index = 0
        self.round_id = 3

        self.final_topics: list[dict[str, Any]] = []
        self.final_topics_by_expert: dict[str, dict[str, Any]] = {}
        self.expert_feature_plans: dict[str, dict[str, Any]] = {}
        self.expert_code_pieces: dict[str, dict[str, Any]] = {}
        self.expert_feature_mapping: list[dict[str, Any]] = []
        self.mapping_by_expert: dict[str, list[dict[str, Any]]] = {}
        self.expert_stats_rows: list[dict[str, Any]] = []
        self.baseline_stats_rows: list[dict[str, Any]] = []
        self.expert_stats_by_idx: dict[int, dict[str, Any]] = {}
        self.baseline_feature_semantics: str = ""
        self.feature_indexing_spec: dict[str, Any] = {}
        self.preflight_report: dict[str, Any] = {}

        self.evaluator_feedback: dict[str, Any] | None = None
        self.expert_corrections: dict[str, dict[str, Any]] = {}
        self.updated_expert_feature_plans: dict[str, dict[str, Any]] = {}
        self.updated_expert_code_pieces: dict[str, dict[str, Any]] = {}
        self.assembler_result: dict[str, Any] | None = None

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config()
        self._load_inputs()
        self._preflight_validate()
        write_yaml(self.output_dir / "preflight_report.yaml", self.preflight_report)

        self._call_evaluator()
        self._show_phase3_object("Evaluator feedback", self.evaluator_feedback)
        self._pause("Review evaluator feedback. Press ENTER to apply expert corrections.")

        self._apply_expert_corrections()
        self._show_phase3_object("Expert corrections", self._redact_code_fields(self.expert_corrections))
        self._pause("Review expert corrections. Press ENTER to assemble the phase-3 feature function.")

        result = self._assemble_feature_function()
        self._show_phase3_object("Phase-3 assembler result", self._redact_code_fields(result))
        self._pause("Review phase-3 assembler result. Press ENTER to finish phase 3.")
        return result

    def _load_inputs(self) -> None:
        required = [
            self.phase12_run_dir / "final_topics.yaml",
            self.phase12_run_dir / "expert_feature_plans.yaml",
            self.phase12_run_dir / "expert_code_pieces.yaml",
            self.phase12_run_dir / "expert_feature_mapping.yaml",
            self.expert_stats_csv,
            self.baseline_stats_csv,
            self.baseline_feature_description,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing phase-3 required input files: " + ", ".join(missing))

        self.final_topics = load_yaml(self.phase12_run_dir / "final_topics.yaml") or []
        if not isinstance(self.final_topics, list):
            raise ValueError("final_topics.yaml must contain a list.")
        self.final_topics_by_expert = {str(t["expert_id"]): t for t in self.final_topics if isinstance(t, dict) and "expert_id" in t}

        self.expert_feature_plans = load_yaml(self.phase12_run_dir / "expert_feature_plans.yaml") or {}
        self.expert_code_pieces = load_yaml(self.phase12_run_dir / "expert_code_pieces.yaml") or {}
        self.expert_feature_mapping = load_yaml(self.phase12_run_dir / "expert_feature_mapping.yaml") or []
        if not isinstance(self.expert_feature_plans, dict):
            raise ValueError("expert_feature_plans.yaml must contain a mapping keyed by expert_id.")
        if not isinstance(self.expert_code_pieces, dict):
            raise ValueError("expert_code_pieces.yaml must contain a mapping keyed by expert_id.")
        if not isinstance(self.expert_feature_mapping, list):
            raise ValueError("expert_feature_mapping.yaml must contain a list.")

        self.expert_stats_rows = self._read_stats_csv(self.expert_stats_csv)
        self.baseline_stats_rows = self._read_stats_csv(self.baseline_stats_csv)
        self.expert_stats_by_idx = {int(r["idx"]): r for r in self.expert_stats_rows}
        self.baseline_feature_semantics = self._read_description(self.baseline_feature_description)

        mapping_by_expert: dict[str, list[dict[str, Any]]] = {}
        for item in self.expert_feature_mapping:
            if isinstance(item, dict) and isinstance(item.get("expert_id"), str):
                mapping_by_expert.setdefault(item["expert_id"], []).append(item)
        self.mapping_by_expert = mapping_by_expert

    def _read_stats_csv(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row: dict[str, Any] = {}
                for key, value in raw.items():
                    if value is None:
                        row[key] = None
                        continue
                    if key in {"tls_id"}:
                        row[key] = value
                    elif key in {"idx", "step", "nan", "inf", "n_samples"}:
                        row[key] = int(float(value)) if str(value).strip() else 0
                    else:
                        row[key] = float(value) if str(value).strip() else 0.0
                rows.append(row)
        if not rows:
            raise ValueError(f"Statistics CSV is empty: {path}")
        if "idx" not in rows[0]:
            raise ValueError(f"Statistics CSV must contain idx column: {path}")
        return rows

    def _read_description(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        try:
            obj = yaml.safe_load(text)
            if obj is not None:
                return dump_yaml(obj)
        except Exception:
            pass
        return text

    def _preflight_validate(self) -> None:
        expert_dim = max(int(r["idx"]) for r in self.expert_stats_rows) + 1
        baseline_dim = max(int(r["idx"]) for r in self.baseline_stats_rows) + 1
        expert_ids_from_mapping = sorted(self.mapping_by_expert)
        expert_ids_from_topics = sorted(self.final_topics_by_expert)
        expert_ids_from_plans = sorted(self.expert_feature_plans)
        expert_ids_from_code = sorted(self.expert_code_pieces)

        errors: list[str] = []
        warnings: list[str] = []
        occupied: dict[int, str] = {}

        for item_i, item in enumerate(self.expert_feature_mapping):
            if not isinstance(item, dict):
                errors.append(f"expert_feature_mapping[{item_i}] is not a mapping.")
                continue
            expert_id = item.get("expert_id")
            if not isinstance(expert_id, str) or not expert_id.strip():
                errors.append(f"expert_feature_mapping[{item_i}] has invalid expert_id.")
                continue
            indices = item.get("feature_indices")
            if not isinstance(indices, dict):
                errors.append(f"{expert_id}: feature_indices must be a mapping.")
                continue
            try:
                start = int(indices["start"])
                end = int(indices["end"])
            except Exception:
                errors.append(f"{expert_id}: feature_indices.start/end must be integers.")
                continue
            if start < 0 or end <= start:
                errors.append(f"{expert_id}: invalid feature range [{start}, {end}).")
                continue
            if end > expert_dim:
                errors.append(f"{expert_id}: range end {end} exceeds expert stats dim {expert_dim}.")
                continue
            for idx in range(start, end):
                if idx in occupied:
                    errors.append(f"feature index {idx} overlaps between {occupied[idx]} and {expert_id}.")
                occupied[idx] = expert_id

        for expert_id in expert_ids_from_mapping:
            if expert_id not in self.final_topics_by_expert:
                errors.append(f"{expert_id}: present in mapping but absent from final_topics.yaml.")
            if expert_id not in self.expert_feature_plans:
                errors.append(f"{expert_id}: present in mapping but absent from expert_feature_plans.yaml.")
            if expert_id not in self.expert_code_pieces:
                errors.append(f"{expert_id}: present in mapping but absent from expert_code_pieces.yaml.")

        missing_indices = sorted(set(range(expert_dim)) - set(occupied))
        if missing_indices:
            warnings.append(
                f"expert feature mapping does not cover {len(missing_indices)} expert-local dimensions; "
                f"first missing indices: {missing_indices[:20]}"
            )

        self.feature_indexing_spec = {
            "expert_feature_stats_index_base": "expert_feature_vector_local",
            "baseline_feature_stats_index_base": "baseline_or_core_feature_vector_local",
            "expert_feature_mapping_index_base": "expert_feature_vector_local",
            "expert_feature_dim": expert_dim,
            "baseline_feature_dim": baseline_dim,
            "expert_stats_csv": str(self.expert_stats_csv),
            "baseline_stats_csv": str(self.baseline_stats_csv),
            "notes": [
                "Expert feature indices are local to the expert add-on feature vector, not the full concatenated state vector.",
                "Baseline feature indices are local to the baseline/core feature vector.",
                "Do not compare baseline indices one-to-one with expert indices unless semantics explicitly justify it.",
            ],
        }

        self.preflight_report = {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "expert_feature_dim": expert_dim,
            "baseline_feature_dim": baseline_dim,
            "expert_ids_from_mapping": expert_ids_from_mapping,
            "expert_ids_from_topics": expert_ids_from_topics,
            "expert_ids_from_plans": expert_ids_from_plans,
            "expert_ids_from_code": expert_ids_from_code,
            "feature_indexing_spec": self.feature_indexing_spec,
        }
        if errors:
            raise ValueError("Phase-3 preflight validation failed. See preflight_report.yaml after rerun or errors: " + "; ".join(errors))

    def _call_evaluator(self) -> None:
        context = self._base_context() | {
            "feature_indexing_spec_yaml": dump_yaml(self.feature_indexing_spec),
            "final_topics_yaml": dump_yaml(self.final_topics),
            "expert_feature_plans_yaml": dump_yaml(self.expert_feature_plans),
            "expert_feature_mapping_yaml": dump_yaml(self.expert_feature_mapping),
            "expert_feature_stats_yaml": dump_yaml(self.expert_stats_rows),
            "baseline_feature_stats_yaml": dump_yaml(self.baseline_stats_rows),
            "baseline_feature_semantics": self.baseline_feature_semantics,
            "task_specific_statistical_policy": str(self.task_variables.get("task_specific_statistical_policy", "")),
        }
        messages = self.evaluator_templates.render_messages("phase3.evaluator", context)
        call_id, result = self._run_llm_call("evaluator", None, "phase3.evaluator", messages)
        try:
            parsed = parse_yaml_response(result.text)
            validated = self._validate_evaluator_feedback(parsed)
            self.evaluator_feedback = validated
            self._log_success(call_id, "evaluator", None, "phase3.evaluator", messages, result.text, validated)
            write_yaml(self.output_dir / "evaluator_feedback.yaml", validated)
        except Exception as exc:
            self._log_failure(call_id, "evaluator", None, "phase3.evaluator", messages, result.text, exc)
            raise

    def _validate_evaluator_feedback(self, obj: Any) -> dict[str, Any]:
        if not isinstance(obj, dict):
            raise ValueError("Evaluator output must be a YAML mapping.")
        root = obj.get("phase3_evaluator_feedback")
        if not isinstance(root, dict):
            raise ValueError("Evaluator output must contain mapping field: phase3_evaluator_feedback.")
        feedback = root.get("expert_feedback")
        if not isinstance(feedback, list):
            raise ValueError("phase3_evaluator_feedback must contain list field: expert_feedback.")
        expected_ids = set(self.mapping_by_expert)
        seen: set[str] = set()
        for i, item in enumerate(feedback):
            if not isinstance(item, dict):
                raise ValueError(f"expert_feedback[{i}] must be a mapping.")
            expert_id = item.get("expert_id")
            action = item.get("action")
            if not isinstance(expert_id, str) or not expert_id.strip():
                raise ValueError(f"expert_feedback[{i}] must contain expert_id.")
            if expert_id in seen:
                raise ValueError(f"Evaluator returned duplicate feedback for {expert_id}.")
            seen.add(expert_id)
            if action not in VALID_EVALUATOR_ACTIONS:
                raise ValueError(f"Evaluator action for {expert_id} must be one of {sorted(VALID_EVALUATOR_ACTIONS)}.")
            if not isinstance(item.get("routed_feature_rows", []), list):
                raise ValueError(f"Evaluator routed_feature_rows for {expert_id} must be a list.")
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        if missing or extra:
            raise ValueError(f"Evaluator feedback expert_id mismatch. missing={missing}, extra={extra}")
        return root

    def _apply_expert_corrections(self) -> None:
        if self.evaluator_feedback is None:
            raise RuntimeError("Evaluator feedback must exist before expert corrections.")
        feedback_items = self.evaluator_feedback.get("expert_feedback", [])
        feedback_by_expert = {item["expert_id"]: item for item in feedback_items}
        updated_plans = deepcopy(self.expert_feature_plans)
        updated_codes = deepcopy(self.expert_code_pieces)
        corrections: dict[str, dict[str, Any]] = {}

        for expert_id in sorted(self.mapping_by_expert):
            feedback = feedback_by_expert[expert_id]
            action = feedback["action"]
            if action == "no_change":
                corrections[expert_id] = {
                    "expert_id": expert_id,
                    "evaluator_action": action,
                    "correction_decision": "no_change",
                    "reuse_previous_code": True,
                    "skipped_prompt": True,
                    "no_change_notes": "Evaluator requested no correction; previous feature plan and code are reused.",
                }
                continue

            template_key = (
                "phase3.expert_stats_update" if action == "stats_update" else "phase3.expert_feature_plan_update"
            )
            context = self._expert_context(expert_id, feedback)
            messages = self.expert_templates.render_messages(template_key, context)
            call_id, result = self._run_llm_call("expert_phase3_correction", expert_id, template_key, messages)
            try:
                parsed = parse_yaml_response(result.text)
                correction = self._validate_expert_correction(parsed, expert_id, allowed_from_evaluator=action)
                corrections[expert_id] = correction
                self._log_success(call_id, "expert_phase3_correction", expert_id, template_key, messages, result.text, correction)
            except Exception as exc:
                self._log_failure(call_id, "expert_phase3_correction", expert_id, template_key, messages, result.text, exc)
                raise

            decision = correction["correction_decision"]
            if decision == "no_change" or bool(correction.get("reuse_previous_code", False)):
                continue
            updated_code = correction.get("updated_expert_feature_code")
            if decision == "stats_only_code_update":
                updated_codes[expert_id] = self._updated_code_record(expert_id, updated_code, correction)
            elif decision == "feature_plan_and_code_update":
                updated_plan = correction.get("updated_feature_plan")
                updated_plans[expert_id] = self._updated_plan_record(expert_id, updated_plan, correction)
                updated_codes[expert_id] = self._updated_code_record(expert_id, updated_code, correction)
                # Keep the full trace artifact internally consistent even though
                # the assembler-facing view later drops nested feature_plan fields.
                updated_codes[expert_id]["feature_plan"] = updated_plans[expert_id]

        self.expert_corrections = corrections
        self.updated_expert_feature_plans = updated_plans
        self.updated_expert_code_pieces = updated_codes
        write_yaml(self.output_dir / "expert_corrections.yaml", corrections)
        write_yaml(self.output_dir / "updated_expert_feature_plans.yaml", updated_plans)
        write_yaml(self.output_dir / "updated_expert_code_pieces.yaml", updated_codes)

    def _expert_context(self, expert_id: str, feedback: dict[str, Any]) -> dict[str, Any]:
        return self._base_context() | {
            "expert_id": expert_id,
            "current_topic_yaml": dump_yaml(self.final_topics_by_expert.get(expert_id, {})),
            "previous_feature_plan_yaml": dump_yaml(self.expert_feature_plans.get(expert_id, {})),
            "previous_expert_code_yaml": dump_yaml(self.expert_code_pieces.get(expert_id, {})),
            "expert_feature_mapping_slice_yaml": dump_yaml(self.mapping_by_expert.get(expert_id, [])),
            "evaluator_feedback_yaml": dump_yaml(feedback),
            "expert_feature_stats_yaml": dump_yaml(self._expert_rows_for_expert(expert_id)),
            "baseline_feature_stats_yaml": dump_yaml(self.baseline_stats_rows),
            "feature_indexing_spec_yaml": dump_yaml(self.feature_indexing_spec),
            "task_specific_phase3_policy": str(self.task_variables.get("task_specific_phase3_policy", "")),
        }

    def _expert_rows_for_expert(self, expert_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.mapping_by_expert.get(expert_id, []):
            indices = item.get("feature_indices", {})
            start = int(indices.get("start", 0))
            end = int(indices.get("end", 0))
            for idx in range(start, end):
                if idx in self.expert_stats_by_idx:
                    rows.append(self.expert_stats_by_idx[idx])
        return rows

    def _validate_expert_correction(
        self,
        obj: Any,
        expert_id: str,
        *,
        allowed_from_evaluator: str,
    ) -> dict[str, Any]:
        if not isinstance(obj, dict):
            raise ValueError("Expert correction output must be a YAML mapping.")
        if obj.get("expert_id") != expert_id:
            raise ValueError(f"Expert correction expert_id mismatch: expected {expert_id}, got {obj.get('expert_id')!r}.")
        decision = obj.get("correction_decision")
        if decision not in VALID_EXPERT_DECISIONS:
            raise ValueError(f"Invalid correction_decision for {expert_id}: {decision!r}.")
        if allowed_from_evaluator == "stats_update" and decision == "feature_plan_and_code_update":
            raise ValueError(f"{expert_id}: stats_update prompt cannot return feature_plan_and_code_update.")
        if decision == "no_change":
            if obj.get("reuse_previous_code") is not True:
                raise ValueError(f"{expert_id}: no_change must set reuse_previous_code: true.")
            obj.setdefault("updated_feature_plan", None)
            obj.setdefault("updated_expert_feature_code", None)
            return obj
        if obj.get("reuse_previous_code") is True:
            raise ValueError(f"{expert_id}: non-no_change correction cannot set reuse_previous_code: true.")
        updated_code = obj.get("updated_expert_feature_code")
        if not isinstance(updated_code, dict):
            raise ValueError(f"{expert_id}: correction must contain updated_expert_feature_code mapping.")
        validate_expert_feature_code(updated_code)
        if decision == "stats_only_code_update":
            if obj.get("updated_feature_plan") is not None:
                raise ValueError(f"{expert_id}: stats_only_code_update must keep updated_feature_plan null.")
            return obj
        updated_plan = obj.get("updated_feature_plan")
        self._validate_feature_plan_payload(updated_plan, expert_id)
        return obj

    def _validate_feature_plan_payload(self, plan: Any, expert_id: str) -> None:
        if not isinstance(plan, dict):
            raise ValueError(f"{expert_id}: updated_feature_plan must be a mapping.")
        required = [
            "feature_family_name",
            "responsible_topic_application",
            "referenced_approaches",
            "design_rationale",
            "required_observables",
            "computation_strategy",
            "expected_outputs",
        ]
        missing = [key for key in required if key not in plan]
        if missing:
            raise ValueError(f"{expert_id}: updated_feature_plan missing required fields: {missing}")

    def _updated_plan_record(self, expert_id: str, updated_plan: dict[str, Any], correction: dict[str, Any]) -> dict[str, Any]:
        old = deepcopy(self.expert_feature_plans[expert_id])
        old["response"] = {"feature_plan": updated_plan}
        old["phase3_correction"] = self._correction_summary(correction)
        return old

    def _updated_code_record(self, expert_id: str, updated_code: dict[str, Any], correction: dict[str, Any]) -> dict[str, Any]:
        old = deepcopy(self.expert_code_pieces[expert_id])
        old["response"] = updated_code
        old["phase3_correction"] = self._correction_summary(correction)
        if expert_id in self.updated_expert_feature_plans:
            old["feature_plan"] = self.updated_expert_feature_plans[expert_id]
        return old

    def _correction_summary(self, correction: dict[str, Any]) -> dict[str, Any]:
        return {
            "correction_decision": correction.get("correction_decision"),
            "reuse_previous_code": correction.get("reuse_previous_code"),
            "self_critique": correction.get("self_critique"),
            "no_change_notes": correction.get("no_change_notes"),
        }

    def _assemble_feature_function(self) -> dict[str, Any]:
        if not self.updated_expert_feature_plans or not self.updated_expert_code_pieces:
            raise RuntimeError("Updated expert artifacts must exist before assembly.")
        context = self._base_context() | {
            "expert_topic_mapping_yaml": dump_yaml(self.final_topics),
            "expert_feature_plans_yaml": dump_yaml(self._expert_feature_plans_for_assembler()),
            "expert_code_pieces_yaml": dump_yaml(self._expert_code_pieces_for_assembler()),
        }
        messages = self.assembler_templates.render_messages("assembler.assemble_feature_function", context)
        call_id, result = self._run_llm_call("assembler", None, "assembler.assemble_feature_function", messages, tools=self._assembler_tools())
        try:
            parsed = validate_assembler_result(parse_yaml_response(result.text))
            self.assembler_result = parsed
            self._log_success(call_id, "assembler", None, "assembler.assemble_feature_function", messages, result.text, parsed)
            write_yaml(self.output_dir / "assembler_result.yaml", parsed)
            write_text(self.output_dir / "assembled_feature_function.py", parsed["feature_engineering_function_code"])
            write_yaml(self.output_dir / "expert_feature_mapping.yaml", parsed["expert_feature_mapping"])
            return parsed
        except Exception as exc:
            self._log_failure(call_id, "assembler", None, "assembler.assemble_feature_function", messages, result.text, exc)
            raise

    def _expert_feature_plans_for_assembler(self) -> dict[str, dict[str, Any]]:
        return {
            expert_id: {"expert_id": expert_id, "response": self._drop_assembler_duplicate_fields(record.get("response", record))}
            for expert_id, record in self.updated_expert_feature_plans.items()
        }

    def _expert_code_pieces_for_assembler(self) -> dict[str, dict[str, Any]]:
        return {
            expert_id: {"expert_id": expert_id, "response": self._drop_assembler_duplicate_fields(record.get("response", record))}
            for expert_id, record in self.updated_expert_code_pieces.items()
        }

    def _drop_assembler_duplicate_fields(self, obj: Any) -> Any:
        duplicate_keys = {
            "topic",
            "current_topic",
            "final_topic",
            "feature_plan",
            "scenario_feature_plan",
            "external_reference_context",
            "external_reference_context_yaml",
            "external_references",
            "retrieved_context",
            "retrieved_references",
            "reference_context",
            "reference_material",
            "referenced_approaches",
            "references",
            "sources",
            "phase3_correction",
        }
        if isinstance(obj, dict):
            return {key: self._drop_assembler_duplicate_fields(value) for key, value in obj.items() if key not in duplicate_keys}
        if isinstance(obj, list):
            return [self._drop_assembler_duplicate_fields(value) for value in obj]
        return obj

    def _base_context(self) -> dict[str, Any]:
        context = dict(self.task_variables)
        context.setdefault("external_reference_context_yaml", "[]")
        context.setdefault("task_specific_statistical_policy", "")
        context.setdefault("task_specific_phase3_policy", "")
        return context

    def _assembler_tools(self) -> list[dict[str, Any]] | None:
        if not self.enable_assembler_web_search:
            return None
        return [{"type": "web_search"}]

    def _run_llm_call(
        self,
        role: str,
        expert_id: str | None,
        template_key: str,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, Any]:
        self.call_index += 1
        call_id = f"call_{self.call_index:04d}_{uuid4().hex[:8]}"
        write_text(self.output_dir / "rendered_prompts" / f"{call_id}.txt", self._format_messages(messages))
        result = self.llm.generate(messages, tools=tools)  # type: ignore[call-arg]
        write_text(self.output_dir / "raw_responses" / f"{call_id}.txt", result.text)
        return call_id, result

    def _log_success(
        self,
        call_id: str,
        role: str,
        expert_id: str | None,
        template_key: str,
        messages: list[Message],
        response_text: str,
        parsed: Any,
    ) -> None:
        write_yaml(self.output_dir / "parsed_responses" / f"{call_id}.yaml", parsed)
        log = LLMCallLog(
            call_id=call_id,
            role=role,
            expert_id=expert_id,
            round_id=self.round_id,
            template_key=template_key,
            messages=[m.to_dict() for m in messages],
            response_text=response_text,
            parsed=parsed,
        )
        append_jsonl(self.output_dir / "calls.jsonl", log.asdict())

    def _log_failure(
        self,
        call_id: str,
        role: str,
        expert_id: str | None,
        template_key: str,
        messages: list[Message],
        response_text: str,
        exc: Exception,
    ) -> None:
        failure = {
            "call_id": call_id,
            "role": role,
            "expert_id": expert_id,
            "template_key": template_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_yaml(self.output_dir / "failed_raw_responses" / f"{call_id}_error.yaml", failure)
        write_text(self.output_dir / "failed_raw_responses" / f"{call_id}_response.txt", response_text)
        append_jsonl(self.output_dir / "failed_calls.jsonl", failure)

    def _write_run_config(self) -> None:
        write_yaml(
            self.output_dir / "run_config.yaml",
            {
                "task_id": self.task_id,
                "phase": "phase3_one_shot_statistical_correction",
                "phase12_run_dir": str(self.phase12_run_dir),
                "expert_stats_csv": str(self.expert_stats_csv),
                "baseline_stats_csv": str(self.baseline_stats_csv),
                "baseline_feature_description": str(self.baseline_feature_description),
                "evaluator_template_path": str(self.evaluator_template_path),
                "expert_template_path": str(self.expert_template_path),
                "assembler_template_path": str(self.assembler_template_path),
                "task_variables_path": str(self.task_variables_path),
                "interactive": self.interactive,
                "enable_assembler_web_search": self.enable_assembler_web_search,
            },
        )

    def _show_phase3_object(self, title: str, obj: Any) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(dump_yaml(obj))

    def _pause(self, prompt: str) -> None:
        if self.interactive:
            input(f"\n{prompt}\n")

    def _format_messages(self, messages: list[Message]) -> str:
        return "\n\n".join(f"[{m.role}]\n{m.content}" for m in messages)

    def _redact_code_fields(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            redacted = {}
            for key, value in obj.items():
                if key in {"feature_code", "feature_engineering_function_code", "code"} and isinstance(value, str):
                    redacted[key] = f"<omitted {len(value.splitlines())} lines>"
                else:
                    redacted[key] = self._redact_code_fields(value)
            return redacted
        if isinstance(obj, list):
            return [self._redact_code_fields(x) for x in obj]
        return obj
