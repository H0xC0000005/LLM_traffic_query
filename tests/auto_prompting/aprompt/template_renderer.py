from __future__ import annotations

from copy import deepcopy
from typing import Any

from jinja2 import Environment, StrictUndefined

from .models import Message


class TemplateStore:
    def __init__(self, template_yaml: dict[str, Any]):
        self.template_yaml = template_yaml
        self.env = Environment(undefined=StrictUndefined, autoescape=False)

    def get_template(self, template_key: str) -> dict[str, Any]:
        node: Any = self.template_yaml.get("templates", {})
        for part in template_key.split("."):
            node = node[part]
        return deepcopy(node)

    def render_messages(self, template_key: str, context: dict[str, Any]) -> list[Message]:
        template = self.get_template(template_key)
        rendered = self._render_obj(template, context)
        messages = rendered["messages"]
        return [Message(role=m["role"], content=m["content"]) for m in messages]

    def _render_obj(self, obj: Any, context: dict[str, Any]) -> Any:
        if isinstance(obj, str):
            return self.env.from_string(obj).render(**context)
        if isinstance(obj, list):
            return [self._render_obj(x, context) for x in obj]
        if isinstance(obj, dict):
            return {k: self._render_obj(v, context) for k, v in obj.items()}
        return obj
