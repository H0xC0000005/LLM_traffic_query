from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Role = Literal["system", "developer", "user", "assistant"]
Action = Literal["keep", "refine", "switch"]


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class TopicRecord:
    expert_id: str
    round_id: int
    aspect: str
    aspect_summary: str
    status: str = "active"

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "aspect": self.aspect,
            "aspect_summary": self.aspect_summary,
        }


@dataclass
class ExpertState:
    expert_id: str
    frozen: bool = False
    current_topic: TopicRecord | None = None
    history: list[Message] = field(default_factory=list)
    topic_history: list[TopicRecord] = field(default_factory=list)

    def set_topic(self, topic: TopicRecord) -> None:
        if self.current_topic is not None:
            self.current_topic.status = "replaced"
        self.current_topic = topic
        self.topic_history.append(topic)


@dataclass
class JudgeDecision:
    expert_id: str
    action: Action
    cluster_name: str | None = None
    reason: str | None = None


@dataclass
class LLMResult:
    text: str
    raw: Any | None = None
    provider: str | None = None
    model: str | None = None


@dataclass
class LLMCallLog:
    call_id: str
    role: str
    expert_id: str | None
    round_id: int
    template_key: str
    messages: list[dict[str, str]]
    response_text: str
    parsed: Any

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
