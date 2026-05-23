from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .io_utils import append_jsonl, dump_yaml, load_yaml, write_text, write_yaml
from .llm import LLMClient
from .models import ExpertState, JudgeDecision, LLMCallLog, Message, TopicRecord
from .parsing import normalize_judge_result, parse_yaml_response, validate_expert_topic
from .template_renderer import TemplateStore


class TopicFormationOrchestrator:
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
        self.max_rounds = max_rounds
        self.interactive = interactive
        self.max_refine_count = max_refine_count
        self.experts = [ExpertState(expert_id=f"expert_{i:02d}") for i in range(1, num_experts + 1)]
        # Max-refine enforcement: consecutive refine counts are code-owned action state.
        self.consecutive_refine_counts = {expert.expert_id: 0 for expert in self.experts}
        self.round_id = 0
        self.call_index = 0
        # Judge memory: compact prior overlap history used to stabilize judge granularity.
        self.judge_memory: list[dict[str, Any]] = []

    def run(self) -> list[TopicRecord]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config()

        self._collect_initial_topics()

        for r in range(1, self.max_rounds + 1):
            self.round_id = r
            self._show_topics(f"Topics after expert update, before judge round {r}")
            self._pause("Review the expert topic set above. Press ENTER to send it to a fresh judge session.")

            no_overlap, clusters, decisions, parsed_judge = self._judge_topics()
            self._show_judge_result(parsed_judge)
            self._pause("Review the judge feedback above. Press ENTER to apply keep/refine/switch actions.")

            if no_overlap:
                print("Judge reports no overlap. Topic formation is complete.")
                break

            changed = self._apply_judge_feedback(clusters, decisions)
            if not changed:
                print("No refine/switch actions were applicable. Stopping to avoid a dead loop.")
                break
        else:
            print(f"Reached max_rounds={self.max_rounds}. Returning current topics.")

        final_topics = [e.current_topic for e in self.experts if e.current_topic is not None]
        write_yaml(self.run_dir / "final_topics.yaml", [asdict(t) for t in final_topics])
        return final_topics

    def _collect_initial_topics(self) -> None:
        for expert in self.experts:
            messages = self.template_store.render_messages("expert.propose_topic", self._base_context())
            expert.history.extend(messages)
            result = self.llm.generate(expert.history)
            expert.history.append(Message(role="assistant", content=result.text))
            parsed = validate_expert_topic(parse_yaml_response(result.text))
            topic = TopicRecord(
                expert_id=expert.expert_id,
                round_id=0,
                aspect=parsed["aspect"],
                aspect_summary=parsed["aspect_summary"],
            )
            expert.set_topic(topic)
            self._log_call(
                role="expert",
                expert_id=expert.expert_id,
                template_key="expert.propose_topic",
                messages=expert.history[:-1],
                response_text=result.text,
                parsed=parsed,
            )

    def _judge_topics(self) -> tuple[bool, list[dict[str, Any]], list[JudgeDecision], Any]:
        topic_collection_yaml = dump_yaml([e.current_topic.to_prompt_dict() for e in self.experts if e.current_topic])
        frozen_topics_yaml = dump_yaml(self._frozen_topic_dicts())
        # Judge memory: pass compact prior overlap history as runtime context.
        judge_memory_yaml = dump_yaml(self._judge_memory_object())
        # Max-refine enforcement: pass current-round action constraints as runtime context.
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
        # Max-refine enforcement: override illegal consecutive refine decisions before applying feedback.
        decisions, max_refine_notes = self._enforce_max_refine_invariants(decisions)
        enforcement_notes.extend(max_refine_notes)
        # Max-refine enforcement: update counters from final code-enforced decisions.
        self._update_consecutive_refine_counts(decisions)
        # Judge memory: update only after code-level keep/max-refine enforcement has produced final decisions.
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

    def _apply_judge_feedback(self, clusters: list[dict[str, Any]], decisions: list[JudgeDecision]) -> bool:
        cluster_by_expert = self._cluster_by_expert(clusters)
        decision_by_expert = {d.expert_id: d for d in decisions}
        changed = False

        for expert in self.experts:
            decision = decision_by_expert.get(expert.expert_id)
            if decision is None:
                continue

            if decision.action == "keep":
                expert.frozen = True
                continue

            if expert.frozen:
                # Code-level invariant: a previously frozen topic cannot be refined or switched.
                # _enforce_keep_invariants should already have converted this to keep.
                continue

            if decision.action not in {"refine", "switch"}:
                continue

            template_key = f"expert.{decision.action}_topic"
            runtime_context = self._runtime_context_for_expert(
                expert, cluster_by_expert.get(expert.expert_id), decision
            )
            messages = self.template_store.render_messages(template_key, self._base_context() | runtime_context)
            expert.history.extend(messages)

            # debug check for expert fed prompts
            # self._review_expert_prompt(
            #     expert=expert,
            #     template_key=template_key,
            #     outgoing_messages=expert.history,
            # )
            result = self.llm.generate(expert.history)
            expert.history.append(Message(role="assistant", content=result.text))
            parsed = validate_expert_topic(parse_yaml_response(result.text))
            topic = TopicRecord(
                expert_id=expert.expert_id,
                round_id=self.round_id,
                aspect=parsed["aspect"],
                aspect_summary=parsed["aspect_summary"],
            )
            expert.set_topic(topic)
            expert.frozen = False
            changed = True
            self._log_call(
                role="expert",
                expert_id=expert.expert_id,
                template_key=template_key,
                messages=expert.history[:-1],
                response_text=result.text,
                parsed=parsed,
            )

        return changed

    def _runtime_context_for_expert(
        self, expert: ExpertState, cluster: dict[str, Any] | None, decision: Any
    ) -> dict[str, Any]:
        if expert.current_topic is None:
            raise ValueError(f"Expert {expert.expert_id} has no current topic.")
        fixed_topics = [
            e.current_topic.to_prompt_dict() for e in self.experts if e.frozen and e.current_topic is not None
        ]
        cluster_obj = cluster or {"cluster_name": decision.cluster_name or "unknown", "members": []}
        current_topic_yaml = dump_yaml(expert.current_topic.to_prompt_dict())
        current_cluster_yaml = dump_yaml(cluster_obj)
        fixed_topics_yaml = dump_yaml(fixed_topics)
        return {
            # New names. Prefer these in templates.
            "current_topic_yaml": current_topic_yaml,
            "current_cluster_yaml": current_cluster_yaml,
            "fixed_topics_yaml": fixed_topics_yaml,
            "judge_instruction": decision.reason or "",
            "action_type": decision.action,
            # Compatibility names for your earlier template.
            "current_topic": current_topic_yaml,
            "current_cluster": current_cluster_yaml,
            "fixed_topics": fixed_topics_yaml,
        }

    def _frozen_topic_dicts(self) -> list[dict[str, Any]]:
        return [e.current_topic.to_prompt_dict() for e in self.experts if e.frozen and e.current_topic is not None]

    # Max-refine enforcement: expose only current-round hard action constraints to the judge.
    def _action_constraints_object(self) -> dict[str, Any]:
        constraints = []
        for expert in self.experts:
            if expert.current_topic is None or expert.frozen:
                continue
            consecutive_refines = self.consecutive_refine_counts.get(expert.expert_id, 0)
            if consecutive_refines >= self.max_refine_count:
                constraints.append(
                    {
                        "expert_id": expert.expert_id,
                        "forbidden_actions": ["refine"],
                        "required_if_not_keep": "switch",
                        "reason": (
                            f"consecutive_refine_count={consecutive_refines} has reached "
                            f"max_refine_count={self.max_refine_count}."
                        ),
                    }
                )
        return {"max_refine_count": self.max_refine_count, "constraints": constraints}

    # Judge memory: frozen topics are intentionally not duplicated here because
    # they are already provided through frozen_topics_yaml.
    def _judge_memory_object(self) -> dict[str, Any]:
        return {"previous_rounds": self.judge_memory}

    # Judge memory: store compact overlap history, not full raw judge transcripts.
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
                members.append(
                    {
                        "expert_id": expert_id,
                        "aspect": member.get("aspect", ""),
                    }
                )

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
            {
                "round_id": self.round_id,
                "no_overlap": no_overlap,
                "overlap_clusters": memory_clusters,
            }
        )

    def _active_expert_ids(self) -> set[str]:
        return {e.expert_id for e in self.experts if e.current_topic is not None}

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
        """Enforce monotonic keep/frozen-topic rules in code.

        Rules:
        1. A previously frozen expert must remain keep.
        2. A current judge keep decision freezes the expert.
        3. Because the judge reports only overlapping clusters, active experts absent from
           overlap clusters are treated as singleton clusters and are frozen as keep.
        """
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
        previous_frozen_ids = {e.expert_id for e in self.experts if e.frozen and e.current_topic is not None}
        singleton_keep_ids = active_ids - cluster_member_ids
        explicit_keep_ids = {d.expert_id for d in decisions if d.action == "keep"}
        forced_keep_ids = previous_frozen_ids | singleton_keep_ids | explicit_keep_ids

        unknown_ids = [d.expert_id for d in decisions if d.expert_id not in active_ids]
        if unknown_ids:
            raise ValueError(f"Judge returned decisions for unknown expert ids: {unknown_ids}")

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

        for expert in self.experts:
            if expert.expert_id in forced_keep_ids:
                expert.frozen = True

        # Under the current schema, clusters means overlapping clusters only.
        no_overlap = len(clusters) == 0
        ordered_decisions = [decision_by_expert[k] for k in sorted(decision_by_expert)]
        return no_overlap, clusters, ordered_decisions, enforcement_notes

    # Max-refine enforcement: final code-owned override after judge parsing.
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

    # Max-refine enforcement: keep counts after final code-enforced judge actions.
    def _update_consecutive_refine_counts(self, decisions: list[JudgeDecision]) -> None:
        for decision in decisions:
            if decision.action == "refine":
                self.consecutive_refine_counts[decision.expert_id] = (
                    self.consecutive_refine_counts.get(decision.expert_id, 0) + 1
                )
            elif decision.action in {"keep", "switch"}:
                self.consecutive_refine_counts[decision.expert_id] = 0

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
            "frozen_expert_ids": sorted(e.expert_id for e in self.experts if e.frozen),
            "max_refine_count": self.max_refine_count,
            "consecutive_refine_counts": dict(sorted(self.consecutive_refine_counts.items())),
            "notes": enforcement_notes,
        }
        # Judge memory: log the compact memory after this judge round for inspection.
        base["judge_memory"] = self._judge_memory_object()
        return base

    def _base_context(self) -> dict[str, Any]:
        return dict(self.task_variables)

    def _cluster_by_expert(self, clusters: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for cluster in clusters:
            for member in cluster.get("members", []):
                expert_id = member.get("expert_id") if isinstance(member, dict) else None
                if isinstance(expert_id, str):
                    result[expert_id] = cluster
        return result

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

    def _write_run_config(self) -> None:
        write_yaml(
            self.run_dir / "run_config.yaml",
            {
                "task_id": self.task_id,
                "template_path": str(self.template_path),
                "task_variables_path": str(self.task_variables_path),
                "num_experts": self.num_experts,
                "max_rounds": self.max_rounds,
                "max_refine_count": self.max_refine_count,
                "interactive": self.interactive,
            },
        )

    def _show_topics(self, title: str) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        topics = [e.current_topic.to_prompt_dict() for e in self.experts if e.current_topic]
        print(dump_yaml(topics))

    def _show_judge_result(self, parsed_judge: Any) -> None:
        print("\n" + "=" * 80)
        print("Judge result")
        print("=" * 80)
        print(dump_yaml(parsed_judge))

    def _review_expert_prompt(
        self,
        expert: ExpertState,
        template_key: str,
        outgoing_messages: list[Message],
    ) -> None:
        if not self.interactive:
            return

        print("\n" + "=" * 80)
        print("Synthesized expert prompt")
        print("=" * 80)
        print(f"Expert: {expert.expert_id}")
        print(f"Template: {template_key}")
        print("-" * 80)
        print(self._format_messages(outgoing_messages))

        self._pause(f"Review the synthesized prompt for {expert.expert_id}. " "Press ENTER to send it to the expert.")

    def _pause(self, prompt: str) -> None:
        if self.interactive:
            input(f"\n{prompt}\n")

    def _format_messages(self, messages: list[Message]) -> str:
        parts = []
        for m in messages:
            parts.append(f"[{m.role}]\n{m.content}")
        return "\n\n".join(parts)
