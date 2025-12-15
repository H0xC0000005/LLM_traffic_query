from pathlib import Path
import yaml
from jinja2 import Environment, BaseLoader

env = Environment(
    loader=BaseLoader,
    autoescape=False,       # we're not rendering HTML
    trim_blocks=True,
    lstrip_blocks=True,
)

def load_prompts(path="prompts.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def render_prompt(cfg, name, **vars):
    # Provide shared fragments to every template by default
    base_vars = {
        "reconsider": cfg["fragments"]["reconsider"],
    }
    tmpl_text = cfg["response_prompt"]["templates"][name]
    return env.from_string(tmpl_text).render({**base_vars, **vars})

cfg = load_prompts()

# Example:
text = render_prompt(
    cfg,
    "bad_reasoning",
    details="you conflated right-of-way with lane priority at the merge",
    rules="- Identify actors\n- Apply local traffic code\n- Output in the requested markdown format"
)
print(text)