# query_handler.py
from __future__ import annotations
import pathlib
import contextlib
import traceback
import sys
import re
import ast
from typing import List, Dict, Optional

import os
from openai import OpenAI
import random

# Import your previously created classes
from prompt_assembler import PromptAssembler, ExternalResolver, read_yaml, dump_yaml
from scene_preprocessor import BasicStatsPreprocessor, FullStatsPreprocessor

# NEW: match files like free_1.yaml, sync_2.yaml, jam_10.yaml, fastphase_3.yaml, etc.
LABEL_NUM_RE = re.compile(r"^([A-Za-z0-9]+)_([0-9]+)\.ya?ml$", re.IGNORECASE)

def list_labeled_scenes(scenes_dir: pathlib.Path) -> list[pathlib.Path]:
    scenes: list[pathlib.Path] = []
    for p in scenes_dir.glob("*.yaml"):
        if LABEL_NUM_RE.match(p.name):
            scenes.append(p)
    return scenes

# def list_numeric_scenes(scenes_dir: pathlib.Path) -> list[pathlib.Path]:
#     items = []
#     for p in scenes_dir.glob("*.yaml"):
#         try:
#             n = int(p.stem)
#         except ValueError:
#             continue
#         items.append((n, p))
#     items.sort(key=lambda t: t[0])
#     return [p for _, p in items]

# ---------------- Utility: simple tee logger ---------------- #

class Tee:
    """Write to both stdout and a file-like."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# ---------------- LLM client (simulated) ---------------- #

class O3Client:
    """
    Thin wrapper around OpenAI Responses API with session chaining via previous_response_id.
    """
    def __init__(self, effort: str = "low"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        assert effort in {"low", "medium", "high"}
        self.effort = effort  # "low" | "medium" | "high"
        self.last_id: str | None = None

    def send(self, prompt: str) -> str:
        kwargs = {
            "model": "o3",
            "reasoning": {"effort": self.effort},
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "You are a traffic scenario analyst. You are expertise in classifying traffic phases "
                                 "according to three-phase traffic theory, based on vehicle data."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
        }
        if self.last_id:
            kwargs["previous_response_id"] = self.last_id

        resp = self.client.responses.create(**kwargs)

        # Save for the next turn in this session
        self.last_id = resp.id

        # Robustly get the model text
        text = getattr(resp, "output_text", None) or ""
        if not text:
            # fallback for older/newer SDK shapes
            try:
                # responses.output[0].content[0].text.value in some SDKs
                text = resp.output[0].content[0].text.value  # type: ignore[attr-defined]
            except Exception:
                text = str(resp)
        return text

# ---------------- Interactive helpers ---------------- #

UNRESOLVED_LIST_RE = re.compile(r"\[(.*?)\]")

def parse_unresolved_names(exc: Exception) -> List[str]:
    """
    Expect message like: "Unresolved placeholders after rendering: ['details', 'foo']"
    We'll extract the [...] and literal_eval it.
    """
    msg = str(exc)
    m = UNRESOLVED_LIST_RE.search(msg)
    if not m:
        return []
    try:
        return list(ast.literal_eval("[" + m.group(1) + "]"))
    except Exception:
        return []

def read_multiline_input(prompt: str) -> str:
    print(prompt)
    print("(Enter your text. Finish with a blank line.)")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()

def prompt_for_kind() -> Optional[str]:
    print("\nEnter LLM response type "
          "[good | incomplete | bad_result | incomplete_reasoning | bad_reasoning | quit]:")
    kind = input("> ").strip().lower()
    if kind in {"good", "incomplete", "bad_result", "incomplete_reasoning", "bad_reasoning"}:
        return kind
    if kind in {"q", "quit", "exit"}:
        return None
    print("Unrecognized type. Try again.")
    return prompt_for_kind()

# ---------------- Core flow ---------------- #

def assemble_response_with_fillins(assembler: PromptAssembler,
                                   kind: str,
                                   overrides: Dict[str, str]) -> str:
    """
    Try rendering the response. If we hit unresolved placeholders,
    ask the user to provide values, stash them into assembler.symbols/overrides, and retry.
    """
    while True:
        try:
            # Pass "details" if present; any other keys we set directly on assembler.symbols
            details = overrides.get("details", "")
            return assembler.render_response(kind, details=details)
        except KeyError as e:
            missing = parse_unresolved_names(e)
            if not missing:
                # Fallback: show error and rethrow if we can't parse names
                print("[ERROR] Rendering failed; could not parse missing placeholders list.")
                raise
            print(f"\n[INFO] Missing placeholders: {missing}")
            for name in missing:
                # If the placeholder is 'details', fill via overrides (since render_response supports it)
                if name == "details":
                    if "details" not in overrides or not overrides["details"]:
                        overrides["details"] = read_multiline_input("Provide value for {{details}}:")
                        print(f"[INFO] Filled {{details}} ({len(overrides['details'])} chars).")
                else:
                    val = read_multiline_input(f"Provide value for {{{{{name}}}}}:".format(name=name))
                    # Persist into assembler's symbol table for future renders in this session
                    assembler.symbols[name] = val
                    print(f"[INFO] Filled {{{{{name}}}}} ({len(val)} chars).".format(name=name))
            # loop and retry

def main():
    project_root = pathlib.Path(__file__).parent
    out_path = project_root / "out.txt"

    # Open out.txt and tee stdout+stderr to it
    with out_path.open("w", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            try:
                print("[SESSION] query_handler starting…")
                template_path = project_root / "LLM_query_template.yaml"
                scenes_dir = project_root / "scenes"

                # Collect all <label>_<n>.yaml scenes (ignores *_example.yaml etc.)
                scene_files_ordered = list_labeled_scenes(scenes_dir)

                # Shuffle for random label ordering; allow optional seeding via env for reproducibility
                seed = os.getenv("LLM_SCENE_SHUFFLE_SEED")
                rng = random.Random(seed) if seed else random
                rng.shuffle(scene_files_ordered)

                scene_ix = 0  # next scene pointer

                # Map known external placeholders to files. Attribute selectors like
                # free_flow_example.vehicles will still resolve via the base name.
                resolver = ExternalResolver(
                    mapping={
                        "free_flow_example": scenes_dir / "free_flow_example.yaml",
                        "synchronized_flow_example": scenes_dir / "synchronized_flow_example.yaml",
                        "wide_moving_jam_example": scenes_dir / "wide_moving_jam_example.yaml",
                    },
                    default_dir=scenes_dir,
                )

                # Build assembler (assumes you've already added dot/arrow attribute support)
                assembler = PromptAssembler(template_path, resolver=resolver, preprocessor=FullStatsPreprocessor(),)

                # Initialize dummy LLM and overrides bag
                llm = llm = O3Client(effort="low")
                overrides: Dict[str, str] = {}

                # ---- Step 1: Send initialization ----
                init_prompt = assembler.render_initialization()
                print("\n[SEND -> LLM] INITIALIZATION PROMPT\n")
                print(init_prompt)
                init_resp = llm.send(init_prompt)
                print(f"\n[DEBUG] response_id: {llm.last_id}\n")
                print("\n[RECV <- LLM] INITIALIZATION RESPONSE (simulated)\n")
                print(init_resp)

                # ---- Step 2: Interactive loop ----
                while True:
                    kind = prompt_for_kind()
                    if kind is None:
                        print("\n[SESSION] Exiting on user request.")
                        break

                    # If the user chose 'good', pick the next scenario and inject it
                    if kind == "good":
                        if scene_ix >= len(scene_files_ordered):
                            print("[WARN] No more numeric scenarios found in scenes_dir.")
                        else:
                            next_scene = scene_files_ordered[scene_ix]
                            print(f">>> sending next scene: {next_scene}")
                            try:
                                # data = read_yaml(next_scene)
                                # assembler.symbols["scenario"] = dump_yaml(data)  # enables {{ scenario }} / {{ scenario.vehicles }}
                                # print(f"[INFO] Injected scenario: {next_scene.name}")
                                # NEW:
                                assembler.set_scene_from_path(next_scene)
                                print(f"[INFO] Injected scenario + precomputed symbols: {next_scene.name}")
                                scene_ix += 1
                            except Exception as e:
                                print(f"[ERROR] Failed to load scenario {next_scene}: {e}")

                    # Build response prompt of chosen type (asks for manual fill-ins when needed)
                    resp_prompt = assemble_response_with_fillins(assembler, kind, overrides)

                    # Send to LLM (simulated), print/record both directions
                    print(f"\n[SEND -> LLM] RESPONSE PROMPT ({kind})\n")
                    print(resp_prompt)
                    print(f">>> sending next scene: {next_scene}\n")

                    llm_reply = llm.send(resp_prompt)
                    print(f"\n[DEBUG] response_id: {llm.last_id}\n")
                    print("\n[RECV <- LLM] RESPONSE (simulated)\n")
                    print(llm_reply)

                print(f"\n[SESSION] Complete. Log written to: {out_path.resolve()}")

            except Exception:
                print("\n[ERROR] Unhandled exception in query_handler:\n")
                traceback.print_exc()

if __name__ == "__main__":
    main()
