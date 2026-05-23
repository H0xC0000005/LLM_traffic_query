from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .io_utils import append_jsonl, dump_yaml, load_yaml, write_text, write_yaml
from .llm import LLMClient
from .models import JudgeDecision, LLMCallLog, Message, TopicRecord
from .parsing import normalize_judge_result, parse_yaml_response, validate_assembler_result
from .template_renderer import TemplateStore


class SingleAgentPhase12Orchestrator:
    """Single-agent ablation orchestrator for phase 1 + phase 2.

    The surrogate agent manages multiple virtual topic slots in one shared history.
    Judge and assembler calls remain fresh-session calls using the same schemas as the
    multi-agent pipeline. The virtual slot IDs intentionally use the existing
    expert_id format so downstream judge, assembler, and feature-mapping schemas remain
    comparable with the main pipeline.
    """

    def __init__(
        self,
        template_path: str | Path,
        task_variables_path: str | Path,
        llm_client: LLMClient,
        run_dir: str | Path,
        num_experts: int,
        max_rounds: int = 5,
        interactive: bool = True,
        max_refine_count: int = 1,
        enable_feature_plan_web_search: bool = False,
        enable_assembler_web_search: bool = False,
    ) -> None:
        self.template_path = Path(template_path)
        self.task_variables_path = Path(task_variables_path)
        self.template_store = TemplateStore(load_yaml(self.template_path))
        task_yaml = load_yaml(self.task_variables_path)
        self.task_variables = dict(task_yaml.get("variables", {}))
        self.task_id = task_yaml.get("task_id", "task")
        self.llm = llm_client
        self.run_dir = Path(run_dir)
        self.num_experts = num_experts
        self.virtual_expert_ids = [f"expert_{i:02d}" for i in range(1, num_experts + 1)]
        self.max_rounds = max_rounds
        self.interactive = interactive
        self.max_refine_count = max_refine_count
        self.enable_feature_plan_web_search = enable_feature_plan_web_search
        self.enable_assembler_web_search = enable_assembler_web_search

        self.single_agent_history: list[Message] = []
        self.topics_by_id: dict[str, TopicRecord] = {}
        self.topic_history_by_id: dict[str, list[TopicRecord]] = {expert_id: [] for expert_id in self.virtual_expert_ids}
        self.frozen_ids: set[str] = set()
        self.consecutive_refine_counts = {expert_id: 0 for expert_id in self.virtual_expert_ids}
        self.judge_memory: list[dict[str, Any]] = []
        self.round_id = 0
        self.call_index = 0

        self.expert_feature_plans: dict[str, dict[str, Any]] = {}
        self.expert_code_pieces: dict[str, dict[str, Any]] = {}
        self.assembler_result: dict[str, Any] | None = None

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config()

        self._propose_initial_topic_set()

        for r in range(1, self.max_rounds + 1):
            self.round_id = r
            self._show_topics(f"Virtual topic slots before judge round {r}")
            self._pause("Review the virtual topic set above. Press ENTER to send it to a fresh judge session.")

            no_overlap, clusters, decisions, parsed_judge = self._judge_topics()
            self._show_phase_object("Judge result", parsed_judge)
            self._pause("Review the judge feedback above. Press ENTER to apply all topic-slot updates.")

            if no_overlap:
                print("Judge reports no overlap. Topic formation is complete.")
                break

            changed = self._apply_judge_feedback_all_at_once(clusters, decisions)
            if not changed:
                print("No refine/switch actions were applicable. Stopping to avoid a dead loop.")
                break
        else:
            print(f"Reached max_rounds={self.max_rounds}. Continuing with current virtual topic set.")

        final_topics = self._topic_records_in_order()
        write_yaml(self.run_dir / "final_topics.yaml", [asdict(t) for t in final_topics])

        self._generate_feature_plans_all_at_once()
        self._show_phase_object("Scenario-specific feature plans", self.expert_feature_plans)
        self._pause("Review the feature plans above. Press ENTER to synthesize all code pieces.")

        self._synthesize_expert_code_all_at_once()
        self._show_phase_object("Expert code pieces without full code blocks", self._redact_code_fields(self.expert_code_pieces))
        self._pause("Review the code metadata above. Press ENTER to send all pieces to the assembler.")

        result = self._assemble_feature_function()
        self._show_phase_object("Assembler result without final code block", self._redact_code_fields(result))
        self._pause("Review the assembler result above. Press ENTER to finish the single-agent ablation run.")
        return result

    def _propose_initial_topic_set(self) -> None:
        context = self._base_context() | {
            "num_experts": self.num_experts,
            "virtual_expert_ids_yaml": dump_yaml(self.virtual_expert_ids),
        }
        messages = self.template_store.render_messages("single_agent.propose_topic_set", context)
        self.single_agent_history.extend(messages)
        result = self.llm.generate(self.single_agent_history)
        self.single_agent_history.append(Message(role="assistant", content=result.text))
        parsed = parse_yaml_response(result.text)
        topics = self._validate_topic_set_output(parsed, round_id=0)
        self.topics_by_id = topics
        for expert_id, topic in topics.items():
            self.topic_history_by_id[expert_id].append(topic)

        self._log_call(
            role="single_agent_topic_set",
            expert_id=None,
            template_key="single_agent.propose_topic_set",
            messages=self.single_agent_history[:-1],
            response_text=result.text,
            parsed=parsed,
        )

    def _judge_topics(self) -> tuple[bool, list[dict[str, Any]], list[JudgeDecision], Any]:
        topic_collection_yaml = dump_yaml(self._topic_collection_dicts())
        frozen_topics_yaml = dump_yaml(self._frozen_topic_dicts())
        judge_memory_yaml = dump_yaml(self._judge_memory_object())
        action_constraints_yaml = dump_yaml(self._action_constraints_object())
        context = self._base_context() | {
            "topic_collection_yaml": topic_collection_yaml,
            "frozen_topics_yaml": frozen_topics_yaml,
            "judge_memory_yaml": judge_memory_yaml,
            "action_constraints_yaml": action_constraints_yaml,
        }
        messages = self.template_store.render_messages("judge.judge_topics", context)
        result = self.llm.generate(messages)
        parsed = parse_yaml_response(result.text)
        no_overlap, clusters, decisions = normalize_judge_result(parsed)
        no_overlap, clusters, decisions, enforcement_notes = self._enforce_keep_invariants(
            no_overlap, clusters, decisions
        )
        decisions, max_refine_notes = self._enforce_max_refine_invariants(decisions)
        enforcement_notes.extend(max_refine_notes)
        self._update_consecutive_refine_counts(decisions)
        self._update_judge_memory(no_overlap, clusters, decisions)
        parsed_for_log = self._judge_log_object(parsed, no_overlap, clusters, decisions, enforcement_notes)
        self._log_call(
            role="judge",
            expert_id=None,
            template_key="judge.judge_topics",
            messages=messages,
            response_text=result.text,
            parsed=parsed_for_log,
        )
        return no_overlap, clusters, decisions, parsed_for_log

    def _apply_judge_feedback_all_at_once(
        self, clusters: list[dict[str, Any]], decisions: list[JudgeDecision]
    ) -> bool:
        update_decisions = [
            d for d in decisions if d.action in {"refine", "switch"} and d.expert_id not in self.frozen_ids
        ]
        update_ids = {d.expert_id for d in update_decisions}
        if not update_ids:
            return False

        context = self._base_context() | {
            "current_topic_collection_yaml": dump_yaml(self._topic_collection_dicts()),
            "clusters_yaml": dump_yaml(clusters),
            "flags_yaml": dump_yaml([asdict(d) for d in decisions]),
            "update_flags_yaml": dump_yaml([asdict(d) for d in update_decisions]),
            "frozen_topics_yaml": dump_yaml(self._frozen_topic_dicts()),
            "fixed_topics_yaml": dump_yaml(self._frozen_topic_dicts()),
            "action_constraints_yaml": dump_yaml(self._action_constraints_object()),
        }
        messages = self.template_store.render_messages("single_agent.update_topic_set", context)
        self.single_agent_history.extend(messages)
        result = self.llm.generate(self.single_agent_history)
        self.single_agent_history.append(Message(role="assistant", content=result.text))
        parsed = parse_yaml_response(result.text)
        updated_topics = self._validate_updated_topics_output(parsed, expected_update_ids=update_ids)

        for expert_id, topic in updated_topics.items():
            self.topics_by_id[expert_id] = topic
            self.topic_history_by_id[expert_id].append(topic)
            self.frozen_ids.discard(expert_id)

        self._log_call(
            role="single_agent_topic_update",
            expert_id=None,
            template_key="single_agent.update_topic_set",
            messages=self.single_agent_history[:-1],
            response_text=result.text,
            parsed=parsed,
        )
        return True

    def _generate_feature_plans_all_at_once(self) -> None:
        context = self._base_context() | {
            "final_topic_collection_yaml": dump_yaml(self._topic_collection_dicts()),
            "virtual_expert_ids_yaml": dump_yaml(self.virtual_expert_ids),
        }
        messages = self.template_store.render_messages("single_agent.apply_topics_to_scenario", context)
        self.single_agent_history.extend(messages)
        result = self.llm.generate(self.single_agent_history, tools=self._feature_plan_tools())
        self.single_agent_history.append(Message(role="assistant", content=result.text))
        parsed = parse_yaml_response(result.text)
        self.expert_feature_plans = self._validate_batch_feature_plans(parsed)
        self._log_call(
            role="single_agent_feature_plans",
            expert_id=None,
            template_key="single_agent.apply_topics_to_scenario",
            messages=self.single_agent_history[:-1],
            response_text=result.text,
            parsed=parsed,
        )
        write_yaml(self.run_dir / "expert_feature_plans.yaml", self.expert_feature_plans)

    def _synthesize_expert_code_all_at_once(self) -> None:
        context = self._base_context() | {
            "virtual_expert_ids_yaml": dump_yaml(self.virtual_expert_ids),
        }
        messages = self.template_store.render_messages("single_agent.synthesize_feature_code", context)
        self.single_agent_history.extend(messages)
        result = self.llm.generate(self.single_agent_history)
        self.single_agent_history.append(Message(role="assistant", content=result.text))
        parsed = parse_yaml_response(result.text)
        self.expert_code_pieces = self._validate_batch_code_pieces(parsed)
        self._log_call(
            role="single_agent_feature_code",
            expert_id=None,
            template_key="single_agent.synthesize_feature_code",
            messages=self.single_agent_history[:-1],
            response_text=result.text,
            parsed=parsed,
        )
        write_yaml(self.run_dir / "expert_code_pieces.yaml", self.expert_code_pieces)

    def _assemble_feature_function(self) -> dict[str, Any]:
        context = self._base_context() | {
            "expert_topic_mapping_yaml": dump_yaml(self._topic_collection_dicts()),
            "expert_feature_plans_yaml": dump_yaml(self._expert_feature_plans_for_assembler()),
            "expert_code_pieces_yaml": dump_yaml(self._expert_code_pieces_for_assembler()),
        }
        messages = self.template_store.render_messages("assembler.assemble_feature_function", context)
        result = self.llm.generate(messages, tools=self._assembler_tools())
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

    def _validate_topic_set_output(self, obj: Any, round_id: int) -> dict[str, TopicRecord]:
        if not isinstance(obj, dict):
            raise ValueError("Single-agent topic-set output must be a YAML mapping.")
        raw_topics = obj.get("topics")
        if not isinstance(raw_topics, list):
            raise ValueError("Single-agent topic-set output must contain list field: topics.")
        result: dict[str, TopicRecord] = {}
        for item in raw_topics:
            if not isinstance(item, dict):
                raise ValueError("Each topic slot must be a mapping.")
            expert_id = item.get("expert_id")
            aspect = item.get("aspect")
            aspect_summary = item.get("aspect_summary")
            if not isinstance(expert_id, str) or not expert_id.strip():
                raise ValueError("Each topic slot must contain expert_id.")
            if expert_id not in self.virtual_expert_ids:
                raise ValueError(f"Unexpected topic slot id: {expert_id!r}.")
            if expert_id in result:
                raise ValueError(f"Duplicate topic slot id: {expert_id}.")
            if not isinstance(aspect, str) or not aspect.strip():
                raise ValueError(f"Topic slot {expert_id} must contain non-empty aspect.")
            if not isinstance(aspect_summary, str) or not aspect_summary.strip():
                raise ValueError(f"Topic slot {expert_id} must contain non-empty aspect_summary.")
            result[expert_id] = TopicRecord(
                expert_id=expert_id,
                round_id=round_id,
                aspect=aspect.strip(),
                aspect_summary=aspect_summary.strip(),
            )
        missing = sorted(set(self.virtual_expert_ids) - set(result))
        extra = sorted(set(result) - set(self.virtual_expert_ids))
        if missing or extra:
            raise ValueError(f"Topic slots do not match expected ids. missing={missing}, extra={extra}")
        return result

    def _validate_updated_topics_output(self, obj: Any, expected_update_ids: set[str]) -> dict[str, TopicRecord]:
        if not isinstance(obj, dict):
            raise ValueError("Single-agent topic-update output must be a YAML mapping.")
        raw_topics = obj.get("updated_topics")
        if not isinstance(raw_topics, list):
            raise ValueError("Single-agent topic-update output must contain list field: updated_topics.")
        result: dict[str, TopicRecord] = {}
        for item in raw_topics:
            if not isinstance(item, dict):
                raise ValueError("Each updated topic must be a mapping.")
            expert_id = item.get("expert_id")
            aspect = item.get("aspect")
            aspect_summary = item.get("aspect_summary")
            if not isinstance(expert_id, str) or not expert_id.strip():
                raise ValueError("Each updated topic must contain expert_id.")
            if expert_id not in expected_update_ids:
                raise ValueError(f"Unexpected updated topic id: {expert_id!r}.")
            if expert_id in result:
                raise ValueError(f"Duplicate updated topic id: {expert_id}.")
            if not isinstance(aspect, str) or not aspect.strip():
                raise ValueError(f"Updated topic {expert_id} must contain non-empty aspect.")
            if not isinstance(aspect_summary, str) or not aspect_summary.strip():
                raise ValueError(f"Updated topic {expert_id} must contain non-empty aspect_summary.")
            result[expert_id] = TopicRecord(
                expert_id=expert_id,
                round_id=self.round_id,
                aspect=aspect.strip(),
                aspect_summary=aspect_summary.strip(),
            )
        missing = sorted(expected_update_ids - set(result))
        if missing:
            raise ValueError(f"Single agent did not update all required topic slots: {missing}")
        return result

    def _validate_batch_feature_plans(self, obj: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(obj, dict):
            raise ValueError("Batch feature-plan output must be a YAML mapping.")
        raw_plans = obj.get("expert_feature_plans")
        if not isinstance(raw_plans, list):
            raise ValueError("Batch feature-plan output must contain list field: expert_feature_plans.")
        required = [
            "feature_family_name",
            "responsible_topic_application",
            "referenced_approaches",
            "design_rationale",
            "required_observables",
            "computation_strategy",
            "expected_outputs",
        ]
        result: dict[str, dict[str, Any]] = {}
        for item in raw_plans:
            if not isinstance(item, dict):
                raise ValueError("Each expert feature plan must be a mapping.")
            expert_id = item.get("expert_id")
            plan = item.get("feature_plan")
            if not isinstance(expert_id, str) or expert_id not in self.topics_by_id:
                raise ValueError(f"Invalid feature-plan expert_id: {expert_id!r}.")
            if expert_id in result:
                raise ValueError(f"Duplicate feature plan for {expert_id}.")
            if not isinstance(plan, dict):
                raise ValueError(f"Feature plan for {expert_id} must contain mapping field: feature_plan.")
            missing = [key for key in required if key not in plan]
            if missing:
                raise ValueError(f"Feature plan for {expert_id} is missing required fields: {missing}")
            result[expert_id] = {"expert_id": expert_id, "response": {"feature_plan": plan}}
        missing_ids = sorted(set(self.virtual_expert_ids) - set(result))
        if missing_ids:
            raise ValueError(f"Missing feature plans for topic slots: {missing_ids}")
        return result

    def _validate_batch_code_pieces(self, obj: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(obj, dict):
            raise ValueError("Batch feature-code output must be a YAML mapping.")
        raw_pieces = obj.get("expert_code_pieces")
        if not isinstance(raw_pieces, list):
            raise ValueError("Batch feature-code output must contain list field: expert_code_pieces.")
        required = [
            "implementation_status",
            "feature_code",
            "feature_outputs",
            "required_inputs",
            "dependencies",
            "unresolved_items",
            "scenario_reflection",
            "alignment_notes",
            "assembler_notes",
        ]
        result: dict[str, dict[str, Any]] = {}
        for item in raw_pieces:
            if not isinstance(item, dict):
                raise ValueError("Each expert code piece must be a mapping.")
            expert_id = item.get("expert_id")
            code_piece = item.get("expert_feature_code")
            if not isinstance(expert_id, str) or expert_id not in self.topics_by_id:
                raise ValueError(f"Invalid code-piece expert_id: {expert_id!r}.")
            if expert_id in result:
                raise ValueError(f"Duplicate code piece for {expert_id}.")
            if not isinstance(code_piece, dict):
                raise ValueError(f"Code piece for {expert_id} must contain mapping field: expert_feature_code.")
            missing = [key for key in required if key not in code_piece]
            if missing:
                raise ValueError(f"Code piece for {expert_id} is missing required fields: {missing}")
            if not isinstance(code_piece.get("feature_code"), str) or not code_piece["feature_code"].strip():
                raise ValueError(f"Code piece for {expert_id} must contain non-empty feature_code.")
            if not isinstance(code_piece.get("feature_outputs"), list):
                raise ValueError(f"Code piece for {expert_id} must contain list field: feature_outputs.")
            result[expert_id] = {"expert_id": expert_id, "response": code_piece}
        missing_ids = sorted(set(self.virtual_expert_ids) - set(result))
        if missing_ids:
            raise ValueError(f"Missing code pieces for topic slots: {missing_ids}")
        return result

    def _topic_collection_dicts(self) -> list[dict[str, Any]]:
        return [self.topics_by_id[expert_id].to_prompt_dict() for expert_id in self.virtual_expert_ids if expert_id in self.topics_by_id]

    def _topic_records_in_order(self) -> list[TopicRecord]:
        return [self.topics_by_id[expert_id] for expert_id in self.virtual_expert_ids if expert_id in self.topics_by_id]

    def _frozen_topic_dicts(self) -> list[dict[str, Any]]:
        return [self.topics_by_id[expert_id].to_prompt_dict() for expert_id in sorted(self.frozen_ids) if expert_id in self.topics_by_id]

    def _active_expert_ids(self) -> set[str]:
        return set(self.topics_by_id)

    def _cluster_member_ids(self, clusters: list[dict[str, Any]]) -> set[str]:
        member_ids: set[str] = set()
        for cluster in clusters:
            members = cluster.get("members", [])
            if not isinstance(members, list):
                continue
            for member in members:
                if isinstance(member, dict) and isinstance(member.get("expert_id"), str):
                    member_ids.add(member["expert_id"])
        return member_ids

    def _enforce_keep_invariants(
        self,
        no_overlap: bool,
        clusters: list[dict[str, Any]],
        decisions: list[JudgeDecision],
    ) -> tuple[bool, list[dict[str, Any]], list[JudgeDecision], list[str]]:
        active_ids = self._active_expert_ids()
        enforcement_notes: list[str] = []
        overlap_clusters: list[dict[str, Any]] = []
        for cluster in clusters:
            member_count = len(
                [
                    member
                    for member in cluster.get("members", [])
                    if isinstance(member, dict) and isinstance(member.get("expert_id"), str)
                ]
            )
            if member_count <= 1:
                enforcement_notes.append(
                    f"Ignored singleton/non-overlap cluster {cluster.get('cluster_name')!r}; treated its topic as singleton keep."
                )
                continue
            overlap_clusters.append(cluster)
        clusters = overlap_clusters
        cluster_member_ids = self._cluster_member_ids(clusters)
        previous_frozen_ids = set(self.frozen_ids)
        singleton_keep_ids = active_ids - cluster_member_ids
        explicit_keep_ids = {d.expert_id for d in decisions if d.action == "keep"}
        forced_keep_ids = previous_frozen_ids | singleton_keep_ids | explicit_keep_ids

        unknown_ids = [d.expert_id for d in decisions if d.expert_id not in active_ids]
        if unknown_ids:
            raise ValueError(f"Judge returned decisions for unknown virtual topic ids: {unknown_ids}")

        decision_by_expert: dict[str, JudgeDecision] = {}
        for decision in decisions:
            if decision.expert_id in decision_by_expert:
                raise ValueError(f"Judge returned duplicate decisions for {decision.expert_id}.")
            if decision.expert_id in forced_keep_ids and decision.action != "keep":
                enforcement_notes.append(
                    f"{decision.expert_id}: judge action {decision.action!r} overridden to 'keep'."
                )
                decision = JudgeDecision(
                    expert_id=decision.expert_id,
                    action="keep",
                    cluster_name=decision.cluster_name,
                    reason=f"Code-enforced keep. Original judge reason: {decision.reason or ''}".strip(),
                )
            decision_by_expert[decision.expert_id] = decision

        for expert_id in sorted(forced_keep_ids - set(decision_by_expert)):
            reasons = []
            if expert_id in previous_frozen_ids:
                reasons.append("previously frozen")
            if expert_id in singleton_keep_ids:
                reasons.append("not present in any overlap cluster; treated as singleton cluster")
            if expert_id in explicit_keep_ids:
                reasons.append("judge assigned keep")
            reason = "Code-enforced keep: " + "; ".join(reasons) + "."
            decision_by_expert[expert_id] = JudgeDecision(expert_id=expert_id, action="keep", reason=reason)
            enforcement_notes.append(f"{expert_id}: added keep decision ({'; '.join(reasons)}).")

        self.frozen_ids.update(forced_keep_ids)
        no_overlap = len(clusters) == 0
        ordered_decisions = [decision_by_expert[k] for k in sorted(decision_by_expert)]
        return no_overlap, clusters, ordered_decisions, enforcement_notes

    def _action_constraints_object(self) -> dict[str, Any]:
        constraints = []
        for expert_id in self.virtual_expert_ids:
            if expert_id not in self.topics_by_id or expert_id in self.frozen_ids:
                continue
            consecutive_refines = self.consecutive_refine_counts.get(expert_id, 0)
            if consecutive_refines >= self.max_refine_count:
                constraints.append(
                    {
                        "expert_id": expert_id,
                        "forbidden_actions": ["refine"],
                        "required_if_not_keep": "switch",
                        "reason": (
                            f"consecutive_refine_count={consecutive_refines} has reached "
                            f"max_refine_count={self.max_refine_count}."
                        ),
                    }
                )
        return {"max_refine_count": self.max_refine_count, "constraints": constraints}

    def _enforce_max_refine_invariants(self, decisions: list[JudgeDecision]) -> tuple[list[JudgeDecision], list[str]]:
        enforcement_notes: list[str] = []
        enforced_decisions: list[JudgeDecision] = []
        for decision in decisions:
            if decision.action != "refine":
                enforced_decisions.append(decision)
                continue
            consecutive_refines = self.consecutive_refine_counts.get(decision.expert_id, 0)
            if consecutive_refines < self.max_refine_count:
                enforced_decisions.append(decision)
                continue
            note = (
                f"{decision.expert_id}: judge action 'refine' overridden to 'switch' "
                f"because consecutive_refine_count={consecutive_refines} reached "
                f"max_refine_count={self.max_refine_count}."
            )
            print(f"Code override: {note}")
            enforcement_notes.append(note)
            enforced_decisions.append(
                JudgeDecision(
                    expert_id=decision.expert_id,
                    action="switch",
                    cluster_name=decision.cluster_name,
                    reason=(
                        "Code-enforced switch due to max_refine_count. "
                        f"Original judge reason: {decision.reason or ''}"
                    ).strip(),
                )
            )
        return enforced_decisions, enforcement_notes

    def _update_consecutive_refine_counts(self, decisions: list[JudgeDecision]) -> None:
        for decision in decisions:
            if decision.action == "refine":
                self.consecutive_refine_counts[decision.expert_id] = (
                    self.consecutive_refine_counts.get(decision.expert_id, 0) + 1
                )
            elif decision.action in {"keep", "switch"}:
                self.consecutive_refine_counts[decision.expert_id] = 0

    def _judge_memory_object(self) -> dict[str, Any]:
        return {"previous_rounds": self.judge_memory}

    def _update_judge_memory(
        self,
        no_overlap: bool,
        clusters: list[dict[str, Any]],
        decisions: list[JudgeDecision],
    ) -> None:
        decision_by_expert = {d.expert_id: d for d in decisions}
        memory_clusters: list[dict[str, Any]] = []
        for cluster in clusters:
            members = []
            for member in cluster.get("members", []):
                if not isinstance(member, dict):
                    continue
                expert_id = member.get("expert_id")
                if not isinstance(expert_id, str):
                    continue
                members.append({"expert_id": expert_id, "aspect": member.get("aspect", "")})
            member_ids = [m["expert_id"] for m in members]
            memory_clusters.append(
                {
                    "cluster_name": cluster.get("cluster_name", ""),
                    "reason": cluster.get("reason", ""),
                    "members": members,
                    "actions": [
                        {
                            "expert_id": expert_id,
                            "action": decision_by_expert[expert_id].action,
                            "reason": decision_by_expert[expert_id].reason or "",
                        }
                        for expert_id in member_ids
                        if expert_id in decision_by_expert
                    ],
                }
            )
        self.judge_memory.append(
            {"round_id": self.round_id, "no_overlap": no_overlap, "overlap_clusters": memory_clusters}
        )

    def _judge_log_object(
        self,
        parsed: Any,
        no_overlap: bool,
        clusters: list[dict[str, Any]],
        decisions: list[JudgeDecision],
        enforcement_notes: list[str],
    ) -> dict[str, Any]:
        base: dict[str, Any]
        if isinstance(parsed, dict):
            base = dict(parsed)
        else:
            base = {"raw_parsed": parsed}
        base["no_overlap"] = no_overlap
        base["clusters"] = clusters
        base["flags"] = [asdict(d) for d in decisions]
        base["code_enforcement"] = {
            "frozen_expert_ids": sorted(self.frozen_ids),
            "max_refine_count": self.max_refine_count,
            "consecutive_refine_counts": dict(sorted(self.consecutive_refine_counts.items())),
            "notes": enforcement_notes,
        }
        base["judge_memory"] = self._judge_memory_object()
        return base

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

    def _feature_plan_tools(self) -> list[dict[str, Any]] | None:
        if not self.enable_feature_plan_web_search:
            return None
        return [{"type": "web_search"}]

    # Assembler API-resolution support: enable web search only for the final
    # assembler call, where simulator/library API details must be resolved.
    def _assembler_tools(self) -> list[dict[str, Any]] | None:
        if not self.enable_assembler_web_search:
            return None
        return [{"type": "web_search"}]

    def _base_context(self) -> dict[str, Any]:
        context = dict(self.task_variables)
        context.setdefault("external_reference_context_yaml", "[]")
        return context

    def _write_run_config(self) -> None:
        write_yaml(
            self.run_dir / "run_config.yaml",
            {
                "task_id": self.task_id,
                "ablation_mode": "single_agent_virtual_slots_phase_1_2",
                "template_path": str(self.template_path),
                "task_variables_path": str(self.task_variables_path),
                "num_experts": self.num_experts,
                "virtual_expert_ids": self.virtual_expert_ids,
                "max_rounds": self.max_rounds,
                "max_refine_count": self.max_refine_count,
                "interactive": self.interactive,
                "enable_feature_plan_web_search": self.enable_feature_plan_web_search,
                "enable_assembler_web_search": self.enable_assembler_web_search,
            },
        )

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

    def _show_topics(self, title: str) -> None:
        self._show_phase_object(title, self._topic_collection_dicts())

    def _show_phase_object(self, title: str, obj: Any) -> None:
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
