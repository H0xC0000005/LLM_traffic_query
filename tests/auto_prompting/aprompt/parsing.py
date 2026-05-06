from __future__ import annotations

from typing import Any

import yaml

from .models import JudgeDecision


VALID_ACTIONS = {"keep", "refine", "switch"}


def strip_yaml_fence(text: str) -> str:
    s = text.strip()
    fence_pairs = [("```", "```"), ("'''", "'''")]
    for open_fence, close_fence in fence_pairs:
        if s.startswith(open_fence):
            lines = s.splitlines()
            if lines and lines[0].strip().lower() in {open_fence, f"{open_fence}yaml", f"{open_fence}yml"}:
                lines = lines[1:]
            if lines and lines[-1].strip() == close_fence:
                lines = lines[:-1]
            return "\n".join(lines).strip()
    return s


def parse_yaml_response(text: str) -> Any:
    return yaml.safe_load(strip_yaml_fence(text))


def validate_expert_topic(obj: Any) -> dict[str, str]:
    if not isinstance(obj, dict):
        raise ValueError("Expert output must be a YAML mapping.")
    aspect = obj.get("aspect")
    aspect_summary = obj.get("aspect_summary")
    if not isinstance(aspect, str) or not aspect.strip():
        raise ValueError("Expert output must contain a non-empty string field: aspect.")
    if not isinstance(aspect_summary, str) or not aspect_summary.strip():
        raise ValueError("Expert output must contain a non-empty string field: aspect_summary.")
    return {"aspect": aspect.strip(), "aspect_summary": aspect_summary.strip()}


def normalize_judge_result(obj: Any) -> tuple[bool, list[dict[str, Any]], list[JudgeDecision]]:
    """Normalize judge output.

    Preferred schema:
      no_overlap: false
      clusters:
        - cluster_name: ...
          reason: ...
          members:
            - expert_id: expert_01
              aspect: ...
              aspect_summary: ...
      flags:
        - expert_id: expert_01
          action: keep/refine/switch
          reason: ...

    A light backward-compatible path accepts:
      clusters: {cluster name: [aspect title, ...]}
      flags: {aspect title: action}
    but old title-keyed flags cannot reliably drive experts unless the titles are unique.
    """
    if not isinstance(obj, dict):
        raise ValueError("Judge output must be a YAML mapping.")

    no_overlap = bool(obj.get("no_overlap", False))
    raw_clusters = obj.get("clusters", []) or []
    raw_flags = obj.get("flags", []) or []

    clusters = _normalize_clusters(raw_clusters)
    decisions = _normalize_flags(raw_flags)

    if not clusters:
        no_overlap = True
    return no_overlap, clusters, decisions


def _normalize_clusters(raw_clusters: Any) -> list[dict[str, Any]]:
    if isinstance(raw_clusters, list):
        clusters: list[dict[str, Any]] = []
        for c in raw_clusters:
            if not isinstance(c, dict):
                raise ValueError("Each cluster must be a mapping.")
            name = c.get("cluster_name") or c.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each cluster must contain cluster_name.")
            members = c.get("members", [])
            if not isinstance(members, list):
                raise ValueError("Cluster members must be a list.")
            clusters.append(
                {
                    "cluster_name": name.strip(),
                    "reason": c.get("reason"),
                    "members": members,
                }
            )
        return clusters

    if isinstance(raw_clusters, dict):
        clusters = []
        for name, members in raw_clusters.items():
            if not isinstance(members, list):
                raise ValueError("Old-style cluster values must be lists.")
            clusters.append(
                {
                    "cluster_name": str(name),
                    "reason": None,
                    "members": [{"aspect": str(x)} for x in members],
                }
            )
        return clusters

    raise ValueError("clusters must be either a list or a mapping.")


def _normalize_flags(raw_flags: Any) -> list[JudgeDecision]:
    if isinstance(raw_flags, list):
        decisions = []
        for f in raw_flags:
            if not isinstance(f, dict):
                raise ValueError("Each flag must be a mapping.")
            expert_id = f.get("expert_id")
            action = str(f.get("action", "")).lower().strip()
            if not isinstance(expert_id, str) or not expert_id.strip():
                raise ValueError("Each flag must contain expert_id.")
            if action not in VALID_ACTIONS:
                raise ValueError(f"Invalid judge action: {action}")
            decisions.append(
                JudgeDecision(
                    expert_id=expert_id.strip(),
                    action=action,  # type: ignore[arg-type]
                    cluster_name=f.get("cluster_name"),
                    reason=f.get("reason"),
                )
            )
        return decisions

    if isinstance(raw_flags, dict):
        decisions = []
        for key, action_raw in raw_flags.items():
            action = str(action_raw).split("#", 1)[0].lower().strip()
            if action not in VALID_ACTIONS:
                raise ValueError(f"Invalid judge action: {action}")
            # Old-style output is title-keyed, not expert-id-keyed. Store it in expert_id for visibility;
            # the orchestrator will only be able to apply it if it matches a real expert id.
            decisions.append(JudgeDecision(expert_id=str(key), action=action))  # type: ignore[arg-type]
        return decisions

    raise ValueError("flags must be either a list or a mapping.")
