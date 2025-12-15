# prompt_assembler.py
from __future__ import annotations
import re
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Tuple
import yaml

# OLD:
# _PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}")

# NEW: capture the whole expression; we'll parse it ourselves
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^\}]+?)\s*\}\}")

def read_yaml(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def dump_yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=False).rstrip()

def is_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool))

def get_by_dotted(root: Any, dotted: str) -> Optional[Any]:
    """
    Traverse the YAML object with a dotted path like: a.b.c
    Returns None if not found.
    """
    node = root
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node

def node_to_text(node: Any) -> str:
    if is_scalar(node):
        return str(node)
    # Serialize structured content as YAML text for insertion
    return dump_yaml(node)

def collect_placeholders(text: str) -> List[str]:
    return list(dict.fromkeys(_PLACEHOLDER_RE.findall(text)))  # unique, stable order

def build_symbol_table(template_yaml: Mapping[str, Any]) -> Dict[str, str]:
    """
    Seed with scalars and convenient keys from 'placeholders'.
    Dotted paths and structured nodes are handled on-demand at render time.
    """
    table: Dict[str, str] = {}

    # Lift 'placeholders' scalars directly
    ph = template_yaml.get("placeholders")
    if isinstance(ph, dict):
        for k, v in ph.items():
            if is_scalar(v):
                table[k] = str(v)
            elif v is not None:
                # If someone puts a list/dict under placeholders, still support it:
                table[k] = dump_yaml(v)

    # (Optional) also lift top-level scalars for convenience
    if isinstance(template_yaml, dict):
        for k, v in template_yaml.items():
            if is_scalar(v):
                table[k] = str(v)

    return table

class ExternalResolver:
    def __init__(self, mapping: Optional[Dict[str, pathlib.Path]] = None,
                 default_dir: Optional[pathlib.Path] = None):
        self.mapping = mapping or {}
        self.default_dir = default_dir

    def maybe_load(self, name: str) -> Optional[str]:
        """
        Try to resolve 'name' from mapping or <default_dir>/<name>.yaml.
        Returns serialized YAML text if found, else None.
        """
        p: Optional[pathlib.Path] = self.mapping.get(name)
        if p is None and self.default_dir is not None:
            candidate = self.default_dir / f"{name}.yaml"
            if candidate.exists():
                p = candidate
        if p is None:
            return None
        data = read_yaml(p)
        return dump_yaml(data)

def render_template(text: str,
                    symbols: Dict[str, str],
                    *,
                    on_missing: Optional[callable] = None,
                    max_sweeps: int = 10) -> str:
    """
    Replace {{ name }} using 'symbols'. If not present, call on_missing(name)
    to try dotted-path or external includes; if provided value is not None,
    cache it into symbols. Raises KeyError with all unresolved names if any remain.
    """
    def replace_once(s: str) -> Tuple[str, bool, List[str]]:
        changed = False
        missing: List[str] = []
        def _sub(m: re.Match) -> str:
            nonlocal changed
            name = m.group(1)
            if name not in symbols:
                if on_missing is not None:
                    val = on_missing(name)
                    if val is not None:
                        symbols[name] = val
                if name not in symbols:
                    # mark missing but leave original placeholder in-place
                    missing.append(name)
                    return m.group(0)
            changed = True
            return symbols[name]
        return _PLACEHOLDER_RE.sub(_sub, s), changed, missing

    out, changed, missing = replace_once(text)
    sweep = 0
    while sweep < max_sweeps:
        if not changed and not missing:
            break
        out, changed, missing = replace_once(out)
        sweep += 1

    # Final check for unresolved names
    unresolved = set(collect_placeholders(out))
    if unresolved:
        raise KeyError(f"Unresolved placeholders after rendering: {sorted(unresolved)}")
    return out

class PromptAssembler:
    def __init__(self, template_path: pathlib.Path,
                resolver: Optional[ExternalResolver] = None,
                vehicles_only_for: Optional[set[str]] = None,
                enable_filters: bool = True):
        self.template_path = template_path
        self.template = read_yaml(template_path)
        self.symbols = build_symbol_table(self.template)
        self.resolver = resolver
        self.vehicles_only_for = set(vehicles_only_for or [])
        self.enable_filters = enable_filters

        self.init_base = self._get_path(["initialization_prompt", "base"])
        self.query_base = self._get_path(["query_prompt", "templates", "query"])
        self.response_templates = self._get_path(["response_prompt", "templates"]) or {}

        # Expose a short alias so {{ query }} resolves inside response templates
        q_node = get_by_dotted(self.template, "query_prompt.templates.query")
        if q_node is None:
            q_node = get_by_dotted(self.template, "query_prompt.templates.base")
        if q_node is not None:
            self.symbols["query"] = node_to_text(q_node)

    def _get_path(self, keys: List[str]) -> Optional[Any]:
        node: Any = self.template
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return node

    def _missing_resolver(self) -> callable:
        def on_missing(name_expr: str) -> Optional[str]:
            # 1) Try exact dotted path in the template (e.g., response_prompt.templates.good)
            node = get_by_dotted(self.template, name_expr.strip())
            if node is not None:
                return node_to_text(node)

            # 2) Split into base + attrs for attribute-style includes
            base, attrs = self._parse_name_expr(name_expr)

            # 2a) Try base inside the template
            node_base = get_by_dotted(self.template, base)
            if node_base is not None:
                val = node_to_text(node_base)
            else:
                # 2b) Try external (e.g., free_flow_example.yaml)
                val = self.resolver.maybe_load(base) if self.resolver is not None else None

            if val is None:
                return None

            # 3) Apply attribute chain (e.g., ".vehicles" or "->vehicles")
            val = self._apply_attr_chain(val, attrs)

            # 4) Keep existing post-processing (e.g., '|' filters, implicit vehicles-only) if you still want it
            return self._postprocess(name_expr, val)
        return on_missing

    def render_initialization(self) -> str:
        if not isinstance(self.init_base, str):
            raise ValueError("initialization_prompt.base must be a string")
        return render_template(self.init_base, dict(self.symbols), on_missing=self._missing_resolver())

    def render_query_for_scene(self, scene_yaml_path: pathlib.Path) -> str:
        if not isinstance(self.query_base, str):
            raise ValueError("query_prompt.templates.base must be a string")
        scene_data = read_yaml(scene_yaml_path)
        scene_yaml_text = dump_yaml(scene_data)
        symbols = dict(self.symbols)
        symbols["scenario"] = scene_yaml_text
        return render_template(self.query_base, symbols, on_missing=self._missing_resolver())

    def render_response(self, kind: str, *, details: str = "") -> str:
        tmpl_map = self.response_templates
        if not isinstance(tmpl_map, dict) or kind not in tmpl_map:
            raise KeyError(f"Unknown response_prompt kind: {kind}")
        tmpl = tmpl_map[kind]
        if not isinstance(tmpl, str):
            raise ValueError(f"response_prompt.templates.{kind} must be a string")
        symbols = dict(self.symbols)
        if details:
            symbols["details"] = details
        return render_template(tmpl, symbols, on_missing=self._missing_resolver())
    
    # NEW: filter application
    def _apply_filters(self, name_expr: str, text: str) -> str:
        if not self.enable_filters:
            return text
        parts = [p.strip() for p in name_expr.split("|")]
        if len(parts) == 1:
            return text
        _, *filters = parts
        out = text
        for f in filters:
            if f == "vehicles":
                try:
                    data = yaml.safe_load(out)
                    if isinstance(data, dict) and "vehicles" in data:
                        out = dump_yaml({"vehicles": data["vehicles"]})
                    elif isinstance(data, list):  # already a list; treat as vehicles
                        out = dump_yaml({"vehicles": data})
                except Exception:
                    pass  # leave as-is if parsing fails
            # (you can add more filters later)
        return out

    # NEW: unified post-processor hook
    def _postprocess(self, name_expr: str, text: str) -> str:
        # 1) explicit filters, if any
        out = self._apply_filters(name_expr, text)
        if out is not text:
            return out
        # 2) implicit rule for specific placeholders
        base = name_expr.split("|", 1)[0].strip()
        if base in self.vehicles_only_for:
            try:
                data = yaml.safe_load(text)
                if isinstance(data, dict) and "vehicles" in data:
                    return dump_yaml({"vehicles": data["vehicles"]})
            except Exception:
                pass
        return text

    
    # NEW: parse "free_flow_example.vehicles" and "free_flow_example -> vehicles"
    def _parse_name_expr(self, name_expr: str) -> tuple[str, list[str]]:
        expr = name_expr.strip()
        # normalize '->' (with or without spaces) into '.'
        expr_norm = re.sub(r"\s*->\s*", ".", expr)
        parts = [p.strip() for p in expr_norm.split(".") if p.strip()]
        if not parts:
            return expr, []
        base, attrs = parts[0], parts[1:]
        return base, attrs

    # NEW: apply attribute chain on YAML text; returns YAML text.
    def _apply_attr_chain(self, text: str, attrs: list[str]) -> str:
        if not attrs:
            return text
        try:
            data = yaml.safe_load(text)
        except Exception:
            # Can't parse—leave as-is
            return text

        node = data
        for a in attrs:
            if isinstance(node, dict) and a in node:
                node = node[a]
            elif isinstance(node, list) and a.isdigit():
                idx = int(a)
                if -len(node) <= idx < len(node):
                    node = node[idx]
                else:
                    return text
            else:
                # Path not found—leave as-is
                return text

        # Emit minimal mapping keyed by the last attribute (so you get "vehicles: [...]")
        return dump_yaml({attrs[-1]: node})

    # Optional helpers for chat-style messages:
    def build_initial_messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.render_initialization()}]

    def build_scene_message(self, scene_yaml_path: pathlib.Path) -> Dict[str, str]:
        return {"role": "user", "content": self.render_query_for_scene(scene_yaml_path)}

    def build_response_message(self, kind: str, *, details: str = "") -> Dict[str, str]:
        return {"role": "user", "content": self.render_response(kind, details=details)}

# ------------------------ Demo / Hardcoded "main" ------------------------ #

def main():
    import pathlib
    import contextlib
    import traceback

    project_root = pathlib.Path(__file__).parent
    out_path = project_root / "out.txt"

    # Redirect both stdout and stderr to the log file.
    with out_path.open("w", encoding="utf-8") as f, \
         contextlib.redirect_stdout(f), \
         contextlib.redirect_stderr(f):
        try:
            template_path = project_root / "LLM_query_template.yaml"

            scenes_dir = project_root / "scene_examples"
            scene_files = [
                scenes_dir / "free_flow_example.yaml",
                scenes_dir / "synchronized_flow_example.yaml",
                scenes_dir / "wide_moving_jam_example.yaml",
            ]

            resolver = ExternalResolver(
                mapping={
                    "free_flow_example": scenes_dir / "free_flow_example.yaml",
                    "synchronized_flow_example": scenes_dir / "synchronized_flow_example.yaml",
                    "wide_moving_jam_example": scenes_dir / "wide_moving_jam_example.yaml",
                },
                default_dir=scenes_dir,
            )

            assembler = PromptAssembler(template_path, resolver=resolver)

            # 1) Initialization
            print("\n====== INITIALIZATION PROMPT ======\n")
            print(assembler.render_initialization())

            # 2) Per-scene queries
            for i, scene_path in enumerate(scene_files, start=1):
                if not scene_path.exists():
                    print(f"\n[WARN] Missing scene file: {scene_path}")
                    continue
                print(f"\n====== SCENE {i} QUERY PROMPT ({scene_path.name}) ======\n")
                print(assembler.render_query_for_scene(scene_path))

            # 3) Example response prompt
            print("\n====== RESPONSE PROMPT (bad_reasoning) ======\n")
            print(assembler.render_response(
                "bad_reasoning",
                details="You ignored density trends and misread the jam propagation direction.",
            ))

            # 4) Chat-style preview
            msgs = assembler.build_initial_messages()
            existing_scene = next((p for p in scene_files if p.exists()), None)
            if existing_scene:
                msgs.append(assembler.build_scene_message(existing_scene))
            print("\n====== CHAT MESSAGES EXAMPLE ======\n")
            for m in msgs:
                body = m["content"]
                print(f"{m['role'].upper()}:\n{body[:400]}{'...' if len(body) > 400 else ''}")
                print("-" * 40)

            print(f"\n[INFO] Wrote output to: {out_path.resolve()}\n")

        except Exception:
            print("\n[ERROR] Unhandled exception:\n")
            traceback.print_exc()

if __name__ == "__main__":
    main()
