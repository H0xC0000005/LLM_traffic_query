"""Run an unchanged Python script with an optional per-process feature module."""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
from pathlib import Path
import runpy
import sys


class FeatureModuleFinder(importlib.abc.MetaPathFinder):
    """Select source at normal import time, preserving the script's import order."""

    def __init__(self, source: str):
        self.source = str(Path(source).resolve())

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "expert_feature_extractor":
            return importlib.util.spec_from_file_location(fullname, self.source)
        return None


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    script = Path(spec["script"]).resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(script.parent))
    if spec.get("extractor"):
        sys.meta_path.insert(0, FeatureModuleFinder(spec["extractor"]))
    sys.argv = [str(script), *spec["argv"]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
