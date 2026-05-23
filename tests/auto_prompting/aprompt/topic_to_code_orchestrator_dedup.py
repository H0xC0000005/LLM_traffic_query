from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from .io_utils import append_jsonl, dump_yaml, write_text, write_yaml
from .llm import LLMClient
from .models import ExpertState, LLMCallLog, Message
from .parsing import (
    parse_yaml_response,
    validate_assembler_result,
    validate_expert_feature_code,
    validate_feature_plan,
)
from .template_renderer import TemplateStore


class TopicToCodeOrchestrator:
    """Procedural phase-2 orchestrator: final topics -> expert code -> assembled function.

    This class consumes the existing ExpertState objects from topic formation so that
    every expert keeps its full local prompt history for later statistical correction.
    """

    def __init__(
        self,
        experts: list[ExpertState],
        template_store: TemplateStore,
        task_variables: dict[str, Any],
        llm_client: LLMClient,
        run_dir: str | Path,
        interactive: bool = True,
        call_index: int = 0,
        enable_feature_plan_web_search: bool = False,
    ) -> None:
        self.experts = experts
        self.template_store = template_store
        self.task_variables = task_variables
        self.llm = llm_client
        self.run_dir = Path(run_dir)
        self.interactive = interactive
        self.call_index = call_index
        self.round_id = 0
        self.expert_feature_plans: dict[str, dict[str, Any]] = {}
        self.expert_code_pieces: dict[str, dict[str, Any]] = {}
        self.assembler_result: dict[str, Any] | None = None
        self.enable_feature_plan_web_search = enable_feature_plan_web_search

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._generate_feature_plans()
        self._show_phase2_object("Scenario-specific feature plans", self.expert_feature_plans)
        self._pause("Review the expert feature plans above. Press ENTER to synthesize expert code pieces.")

        self._synthesize_expert_code()
        self._show_phase2_object("Expert code pieces without full code blocks", self._redact_code_fields(self.expert_code_pieces))
        self._pause("Review the expert code metadata above. Press ENTER to send all pieces to the assembler.")

        result = self._assemble_feature_function()
        self._show_phase2_object("Assembler result without final code block", self._redact_code_fields(result))
        self._pause("Review the assembler result above. Press ENTER to finish phase 2.")
        return result
    
    def _feature_plan_tools(self) -> list[dict[str, Any]] | None:
        if not self.enable_feature_plan_web_search:
            return None
        return [{"type": "web_search"}]

    def _generate_feature_plans(self) -> None:
        for expert in self.experts:
            if expert.current_topic is None:
                raise ValueError(f"Expert {expert.expert_id} has no final topic.")
            context = self._base_context() | {
                "current_topic_yaml": dump_yaml(expert.current_topic.to_prompt_dict()),
            }
            messages = self.template_store.render_messages("expert.apply_topic_to_scenario", context)
            expert.history.extend(messages)
            result = self.llm.generate(expert.history, tools=self._feature_plan_tools())
            expert.history.append(Message(role="assistant", content=result.text))
            parsed = validate_feature_plan(parse_yaml_response(result.text))
            self.expert_feature_plans[expert.expert_id] = {
                "expert_id": expert.expert_id,
                "topic": expert.current_topic.to_prompt_dict(),
                "response": parsed,
            }
            self._log_call(
                role="expert_feature_plan",
                expert_id=expert.expert_id,
                template_key="expert.apply_topic_to_scenario",
                messages=expert.history[:-1],
                response_text=result.text,
                parsed=parsed,
            )

        write_yaml(self.run_dir / "expert_feature_plans.yaml", self.expert_feature_plans)

    def _synthesize_expert_code(self) -> None:
        for expert in self.experts:
            if expert.current_topic is None:
                raise ValueError(f"Expert {expert.expert_id} has no final topic.")
            feature_plan = self.expert_feature_plans.get(expert.expert_id)
            if feature_plan is None:
                raise ValueError(f"Expert {expert.expert_id} has no feature plan.")
            context = self._base_context() | {
                "current_topic_yaml": dump_yaml(expert.current_topic.to_prompt_dict()),
                "feature_plan_yaml": dump_yaml(feature_plan),
            }
            messages = self.template_store.render_messages("expert.synthesize_feature_code", context)
            expert.history.extend(messages)
            result = self.llm.generate(expert.history)
            expert.history.append(Message(role="assistant", content=result.text))
            parsed = validate_expert_feature_code(parse_yaml_response(result.text))
            self.expert_code_pieces[expert.expert_id] = {
                "expert_id": expert.expert_id,
                "topic": expert.current_topic.to_prompt_dict(),
                "feature_plan": feature_plan,
                "response": parsed,
            }
            self._show_phase2_object(
                f"Code synthesis output for {expert.expert_id} without code block",
                self._redact_code_fields(self.expert_code_pieces[expert.expert_id]),
            )
            self._pause(f"Review code synthesis metadata for {expert.expert_id}. Press ENTER to continue.")
            self._log_call(
                role="expert_feature_code",
                expert_id=expert.expert_id,
                template_key="expert.synthesize_feature_code",
                messages=expert.history[:-1],
                response_text=result.text,
                parsed=parsed,
            )

        write_yaml(self.run_dir / "expert_code_pieces.yaml", self.expert_code_pieces)

    def _assemble_feature_function(self) -> dict[str, Any]:
        expert_topic_mapping = [
            expert.current_topic.to_prompt_dict()
            for expert in self.experts
            if expert.current_topic is not None
        ]
        # Assembler input de-duplication: keep full internal artifacts on disk,
        # but provide slimmer assembler-facing views to avoid repeating topic,
        # feature-plan, and external-reference material in the rendered prompt.
        assembler_feature_plans = self._expert_feature_plans_for_assembler()
        assembler_code_pieces = self._expert_code_pieces_for_assembler()
        context = self._base_context() | {
            "expert_topic_mapping_yaml": dump_yaml(expert_topic_mapping),
            "expert_feature_plans_yaml": dump_yaml(assembler_feature_plans),
            "expert_code_pieces_yaml": dump_yaml(assembler_code_pieces),
        }
        messages = self.template_store.render_messages("assembler.assemble_feature_function", context)
        result = self.llm.generate(messages)
        parsed = validate_assembler_result(parse_yaml_response(result.text))
        self.assembler_result = parsed

        write_yaml(self.run_dir / "assembler_result.yaml", parsed)
        write_text(self.run_dir / "assembled_feature_function.py", parsed["feature_engineering_function_code"])
        write_yaml(self.run_dir / "expert_feature_mapping.yaml", parsed["expert_feature_mapping"])
        self._log_call(
            role="assembler",
            expert_id=None,
            template_key="assembler.assemble_feature_function",
            messages=messages,
            response_text=result.text,
            parsed=parsed,
        )
        return parsed


    # Assembler input de-duplication: topic identities are provided once through
    # expert_topic_mapping_yaml, and full feature plans are provided once through
    # expert_feature_plans_yaml. These helpers only slim the assembler prompt;
    # self.expert_feature_plans and self.expert_code_pieces remain full trace data.
    def _expert_feature_plans_for_assembler(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for expert_id, record in self.expert_feature_plans.items():
            result[expert_id] = {
                "expert_id": expert_id,
                "response": self._drop_assembler_duplicate_fields(record.get("response", record)),
            }
        return result

    def _expert_code_pieces_for_assembler(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for expert_id, record in self.expert_code_pieces.items():
            result[expert_id] = {
                "expert_id": expert_id,
                "response": self._drop_assembler_duplicate_fields(record.get("response", record)),
            }
        return result

    def _drop_assembler_duplicate_fields(self, obj: Any) -> Any:
        duplicate_keys = {
            # Topic identity is already supplied by expert_topic_mapping_yaml.
            "topic",
            "current_topic",
            "final_topic",
            # Full feature plans are already supplied by expert_feature_plans_yaml.
            "feature_plan",
            "scenario_feature_plan",
            # External/reference material is useful for expert planning but not
            # needed by the assembler once plan/code outputs have been produced.
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
        }
        if isinstance(obj, dict):
            return {
                key: self._drop_assembler_duplicate_fields(value)
                for key, value in obj.items()
                if key not in duplicate_keys
            }
        if isinstance(obj, list):
            return [self._drop_assembler_duplicate_fields(value) for value in obj]
        return obj

    def _base_context(self) -> dict[str, Any]:
        context = dict(self.task_variables)
        # Safe default for templates that support retrieval/RAG context, while keeping it task-configurable.
        context.setdefault("external_reference_context_yaml", "[]")
        return context

    def _log_call(
        self,
        role: str,
        expert_id: str | None,
        template_key: str,
        messages: list[Message],
        response_text: str,
        parsed: Any,
    ) -> None:
        self.call_index += 1
        call_id = f"call_{self.call_index:04d}_{uuid4().hex[:8]}"
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
        append_jsonl(self.run_dir / "calls.jsonl", log.asdict())
        write_text(self.run_dir / "rendered_prompts" / f"{call_id}.txt", self._format_messages(messages))
        write_text(self.run_dir / "raw_responses" / f"{call_id}.txt", response_text)
        write_yaml(self.run_dir / "parsed_responses" / f"{call_id}.yaml", parsed)

    def _show_phase2_object(self, title: str, obj: Any) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(dump_yaml(obj))

    def _pause(self, prompt: str) -> None:
        if self.interactive:
            input(f"\n{prompt}\n")

    def _format_messages(self, messages: list[Message]) -> str:
        parts = []
        for m in messages:
            parts.append(f"[{m.role}]\n{m.content}")
        return "\n\n".join(parts)

    def _redact_code_fields(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            redacted = {}
            for key, value in obj.items():
                if key in {"feature_code", "feature_engineering_function_code", "code"} and isinstance(value, str):
                    line_count = len(value.splitlines())
                    redacted[key] = f"<omitted {line_count} lines>"
                else:
                    redacted[key] = self._redact_code_fields(value)
            return redacted
        if isinstance(obj, list):
            return [self._redact_code_fields(x) for x in obj]
        return obj
