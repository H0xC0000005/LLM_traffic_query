"""Launch existing experiments from YAML, without importing experiment modules.

The default operation is a read-only plan. Execution requires --run.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import fnmatch
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from uuid import uuid4
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[1]
ENTRY = Path(__file__).with_name("_entry.py")
TOKEN = re.compile(r"\$\{([^{}]+)\}")
FORBIDDEN_FLAGS = {"--use-expert-features", "--use_expert_features"}


class LaunchError(ValueError):
    pass


def read_yaml(path: Path) -> dict:
    # JSON is a YAML subset and keeps fixture tests usable without PyYAML.
    content = path.read_text(encoding="utf-8-sig")
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise LaunchError("YAML configurations require PyYAML in the launcher interpreter.") from exc
        value = yaml.safe_load(content)
    if not isinstance(value, dict):
        raise LaunchError(f"Configuration must be a mapping: {path}")
    return value


def merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def lookup(context: dict, key: str) -> Any:
    value: Any = context
    for part in key.split("."):
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LaunchError(f"Unknown variable: {key}") from exc
    return value


def expand(value: Any, context: dict, depth: int = 0) -> Any:
    if depth > 35:
        raise LaunchError("Variable expansion is circular or too deeply nested.")
    if isinstance(value, dict):
        return {key: expand(item, context, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, context, depth + 1) for item in value]
    if not isinstance(value, str):
        return value
    match = TOKEN.fullmatch(value)
    if match:
        return expand(lookup(context, match[1]), context, depth + 1)
    rendered = TOKEN.sub(lambda m: str(lookup(context, m[1])), value)
    if rendered != value:
        return expand(rendered, context, depth + 1)
    if "${" in rendered:
        raise LaunchError(f"Invalid variable expression: {rendered}")
    return rendered


def references(value: Any) -> set[str]:
    if isinstance(value, dict):
        if "from_job" in value:
            if set(value) != {"from_job", "artifact"}:
                raise LaunchError("Artifact references require exactly from_job and artifact.")
            return {value["from_job"]}
        return set().union(*(references(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(references(v) for v in value)) if value else set()
    return set()


def resolve_references(value: Any, results: dict) -> Any:
    if isinstance(value, dict):
        if "from_job" in value:
            job = value["from_job"]
            try:
                selected = lookup(results[job]["artifacts"], value["artifact"])
            except KeyError as exc:
                raise LaunchError(f"No completed artifacts for {job}") from exc
            if not isinstance(selected, (str, int, float, bool)):
                raise LaunchError(f"Reference {job}:{value['artifact']} must select one scalar value.")
            return selected
        return {key: resolve_references(item, results) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_references(item, results) for item in value]
    return value


def absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (REPO / candidate).resolve()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flag_args(arguments: dict) -> list[str]:
    result: list[str] = []
    for flag, value in arguments.items():
        if flag in FORBIDDEN_FLAGS:
            raise LaunchError(f"Legacy flag is excluded from this launcher: {flag}")
        if not isinstance(flag, str) or not flag.startswith("-"):
            raise LaunchError(f"Expected a CLI flag, got {flag!r}")
        if value is False or value is None:
            continue
        if value is True:
            result.append(flag)
        elif isinstance(value, (str, int, float)):
            result.extend([flag, str(value)])
        else:
            raise LaunchError(f"Argument {flag} requires a scalar value, got {value!r}")
    return result


def declared_flags(script: Path) -> set[str]:
    tree = ast.parse(script.read_text(encoding="utf-8-sig"), filename=str(script))
    return {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-")
    }


def input_record(path: Path, destination: Path | None = None) -> dict:
    if not path.exists():
        raise LaunchError(f"Input does not exist: {path}")
    record = {"path": str(path), "kind": "directory" if path.is_dir() else "file"}
    if path.is_file():
        record.update(sha256=sha256(path), bytes=path.stat().st_size)
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            record["snapshot"] = str(destination)
    return record


def configured_inputs(job: dict) -> list[Any]:
    """Include files named by a supplied SUMO config, without starting SUMO."""
    paths = list(job.get("inputs", []))
    checkpoint = job.get("args", {}).get("--checkpoint")
    if job.get("stage") == "tsc_eval" and isinstance(checkpoint, str):
        metadata = job["args"].get("--meta") or str(absolute(checkpoint).with_suffix(".json"))
        if isinstance(metadata, str) and metadata not in paths:
            paths.append(metadata)
    for entry in list(paths):
        if not isinstance(entry, str):
            continue
        config = absolute(entry)
        if config.suffix != ".sumocfg" or not config.is_file():
            continue
        try:
            root = ET.parse(config).getroot()
        except ET.ParseError as exc:
            raise LaunchError(f"Invalid SUMO configuration XML: {config}") from exc
        for field in ("net-file", "route-files", "additional-files", "weight-files", "load-state"):
            for node in root.findall(f"./input/{field}"):
                for filename in node.get("value", "").split(","):
                    if filename.strip():
                        path = Path(filename.strip())
                        path = path if path.is_absolute() else config.parent / path
                        resolved = str(path.resolve())
                        if resolved not in paths:
                            paths.append(resolved)
    return paths


def yaml_materializations(job: dict, write: bool) -> dict:
    records = {}
    for name, specification in job.get("yaml_inputs", {}).items():
        source = absolute(specification["source"])
        if not source.is_file():
            raise LaunchError(f"YAML input source does not exist: {source}")
        merged = merge(read_yaml(source), specification.get("overrides", {}))
        destination = Path(job["job_dir"]) / "inputs" / f"{name}.yaml"
        if write:
            # JSON is valid YAML and avoids changing any original template file.
            json_write(destination, merged)
        records[name] = {"source": str(source), "sha256": sha256(source),
                         "path": str(destination), "overrides": specification.get("overrides", {})}
    return records


def build_jobs(config: dict, run_dir: Path, local: dict | None = None,
               overrides: dict | None = None) -> dict[str, dict]:
    config = merge(config, local or {})
    if config.get("version") != 1:
        raise LaunchError("Configuration version must be 1.")
    builtins = {"repo": str(REPO), "run_dir": str(run_dir),
                "python_sumo": sys.executable, "python_energy": sys.executable,
                "python_llm": sys.executable, "python_llmfe": sys.executable,
                "sumo_home": os.environ.get("SUMO_HOME", "E:/Sumo")}
    variables = merge(builtins, config.get("variables", {}))
    variables = merge(variables, overrides or {})
    jobs = {}
    for template in config.get("jobs", []):
        matrix = template.get("matrix", {})
        if not isinstance(matrix, dict) or any(not isinstance(v, list) or not v for v in matrix.values()):
            raise LaunchError("Each matrix dimension must be a nonempty list.")
        for row in itertools.product(*matrix.values()):
            context = merge(variables, dict(zip(matrix, row)))
            job_id = expand(template["id"], context)
            if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id) or job_id in {".", ".."}:
                raise LaunchError(f"Unsafe job id: {job_id!r}")
            if job_id in jobs:
                raise LaunchError(f"Duplicate job id: {job_id}")
            context.update(job_id=job_id, job_dir=str(run_dir / "jobs" / job_id))
            raw = merge(config.get("defaults", {}), {k: v for k, v in template.items() if k != "matrix"})
            context["yaml_inputs"] = {name: str(Path(context["job_dir"]) / "inputs" / f"{name}.yaml")
                                      for name in raw.get("yaml_inputs", {})}
            job = expand(raw, context)
            job["id"] = job_id
            job["job_dir"] = context["job_dir"]
            job.setdefault("python", sys.executable)
            job.setdefault("cwd", str(REPO))
            job.setdefault("args", {})
            job.setdefault("outputs", {})
            job.setdefault("env", {})
            job["script"] = str(absolute(job["script"]))
            job["cwd"] = str(absolute(job["cwd"]))
            dependencies = set(job.get("depends_on", [])) | references(job)
            job["depends_on"] = sorted(dependencies)
            jobs[job_id] = job
    if not jobs:
        raise LaunchError("Configuration contains no jobs.")
    for name, job in jobs.items():
        for dependency in job["depends_on"]:
            if dependency not in jobs:
                raise LaunchError(f"{name} references unknown job {dependency}")
        def check_artifacts(value):
            if isinstance(value, dict):
                if "from_job" in value:
                    artifact = value["artifact"].split(".")[0]
                    if artifact not in jobs[value["from_job"]]["outputs"]:
                        raise LaunchError(f"{name} references undeclared artifact {value['from_job']}:{artifact}")
                else:
                    for item in value.values():
                        check_artifacts(item)
            elif isinstance(value, list):
                for item in value:
                    check_artifacts(item)
        check_artifacts(job)
    # Validate the complete graph even when only a subset is selected.
    visited, active = set(), set()
    def visit(name):
        if name in active:
            raise LaunchError(f"Dependency cycle at {name}")
        if name not in visited:
            active.add(name)
            for parent in jobs[name]["depends_on"]:
                visit(parent)
            active.remove(name)
            visited.add(name)
    for name in jobs:
        visit(name)
    return jobs


def select_jobs(jobs: dict, patterns: list[str]) -> dict:
    selected = set()
    for pattern in patterns or ["*"]:
        matches = {name for name in jobs if fnmatch.fnmatchcase(name, pattern)}
        if not matches:
            raise LaunchError(f"Selection matches no jobs: {pattern}")
        selected |= matches
    def include(name):
        for parent in jobs[name]["depends_on"]:
            if parent not in selected:
                selected.add(parent)
                include(parent)
    for name in list(selected):
        include(name)
    return {name: job for name, job in jobs.items() if name in selected}


def preflight(job: dict, execute: bool) -> list[str]:
    warnings = []
    script = Path(job["script"])
    if not script.is_file():
        raise LaunchError(f"Script does not exist: {script}")
    if not Path(job["cwd"]).is_dir():
        raise LaunchError(f"Working directory does not exist: {job['cwd']}")
    python = str(job["python"])
    if not (Path(python).is_file() or shutil.which(python)):
        raise LaunchError(f"Interpreter does not exist: {python}")
    flags = declared_flags(script)
    unknown = set(job["args"]) - flags if flags else set()
    if unknown:
        raise LaunchError(f"{job['id']} uses arguments absent from the script: {sorted(unknown)}")
    for flag, value in job["args"].items():
        if flag in FORBIDDEN_FLAGS:
            raise LaunchError(f"Legacy flag is excluded: {flag}")
        if isinstance(value, dict):
            if execute:
                raise LaunchError(f"Unresolved artifact reference for {flag}")
        else:
            flag_args({flag: value})
    if "__SET_" in json.dumps(job):
        message = "Required external variables are still __SET_ placeholders."
        if execute:
            raise LaunchError(f"{job['id']}: {message}")
        warnings.append(message)
    generated = {record["path"] for record in yaml_materializations(job, False).values()}
    for path in configured_inputs(job):
        if isinstance(path, dict):
            continue
        if str(path) in generated:
            continue
        if not absolute(path).exists():
            message = f"Input does not exist: {path}"
            if execute:
                raise LaunchError(message)
            warnings.append(message)
    extractor = job.get("extractor")
    checkpoint = job["args"].get("--checkpoint")
    if job.get("stage") == "tsc_eval" and isinstance(checkpoint, str):
        metadata = job["args"].get("--meta") or str(absolute(checkpoint).with_suffix(".json"))
        if isinstance(metadata, str) and absolute(metadata).is_file():
            try:
                meta = json.loads(absolute(metadata).read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError) as exc:
                raise LaunchError(f"Cannot read checkpoint metadata: {metadata}") from exc
            if str(meta.get("addon_encoder_name", "")).lower() == "expert" and not extractor:
                message = "Expert checkpoint requires an explicit matching extractor in launch configuration."
                if execute:
                    raise LaunchError(message)
                warnings.append(message)
    if extractor and not isinstance(extractor.get("path"), dict):
        source = absolute(extractor["path"])
        if not source.is_file():
            if execute:
                raise LaunchError(f"Extractor does not exist: {source}")
            warnings.append(f"Extractor does not exist: {source}")
        else:
            symbol = extractor.get("symbol")
            if symbol:
                tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
                if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol for n in tree.body):
                    raise LaunchError(f"Extractor {source} does not define {symbol}")
    if extractor and extractor.get("mapping") and not isinstance(extractor["mapping"], dict):
        if not absolute(extractor["mapping"]).is_file():
            message = f"Extractor mapping does not exist: {extractor['mapping']}"
            if execute:
                raise LaunchError(message)
            warnings.append(message)
    timeout = job.get("wall_timeout_seconds")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise LaunchError("wall_timeout_seconds must be positive.")
    return warnings


def plan_job(job: dict) -> dict:
    argv = []
    for flag, value in job["args"].items():
        if isinstance(value, dict):
            value = f"<artifact:{value['from_job']}:{value['artifact']}>"
        argv.extend(flag_args({flag: value}))
    return {"id": job["id"], "stage": job.get("stage"), "depends_on": job["depends_on"],
            "cwd": job["cwd"], "command": [job["python"], "-u", str(ENTRY), str(Path(job["job_dir"]) / "entry.json")],
            "experiment_command": [job["python"], "-u", job["script"], *argv],
            "entry_spec": {"script": job["script"], "argv": argv,
                           "extractor": str(Path(job["job_dir"]) / "inputs" / "extractor.py") if job.get("extractor") else None},
            "extractor": job.get("extractor"), "outputs": job["outputs"],
            "yaml_inputs": yaml_materializations(job, False), "warnings": preflight(job, False)}


def collect_outputs(job: dict) -> dict:
    import glob
    artifacts = {}
    for name, specification in job["outputs"].items():
        if isinstance(specification, str):
            specification = {"path": specification}
        if "glob" in specification:
            values = sorted(str(Path(p).resolve()) for p in glob.glob(str(absolute(specification["glob"])), recursive=True))
            if specification.get("one") and len(values) > 1:
                raise LaunchError(f"{job['id']}:{name} expected one output, found {len(values)}")
            value = values[0] if specification.get("one") and values else (None if specification.get("one") else values)
            present = bool(values)
        else:
            value = str(absolute(specification["path"]))
            present = Path(value).exists()
        if not present and specification.get("required", True):
            raise LaunchError(f"Missing declared output {job['id']}:{name}")
        artifacts[name] = value if present else None
    return artifacts


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_job(job: dict, cancel: threading.Event | None = None) -> dict:
    preflight(job, True)
    directory = Path(job["job_dir"])
    directory.mkdir(parents=True, exist_ok=False)
    record = {"id": job["id"], "stage": job.get("stage"), "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "artifacts": {}}
    json_write(directory / "status.json", record)
    process = None
    try:
        materialized = yaml_materializations(job, True)
        records = [input_record(Path(job["script"]))]
        for index, source in enumerate(configured_inputs(job)):
            path = absolute(source)
            records.append(input_record(path, directory / "inputs" / f"{index:02d}_{path.name}" if path.is_file() else None))
        extractor = job.get("extractor")
        extractor_snapshot = None
        if extractor:
            source = absolute(extractor["path"])
            extractor_snapshot = directory / "inputs" / "extractor.py"
            records.append(input_record(source, extractor_snapshot))
            if extractor.get("mapping"):
                records.append(input_record(absolute(extractor["mapping"]), directory / "inputs" / "mapping.yaml"))
        argv = flag_args(job["args"])
        launch_spec = {"script": job["script"], "argv": argv,
                       "extractor": str(extractor_snapshot) if extractor_snapshot else None}
        spec_path = directory / "entry.json"
        json_write(spec_path, launch_spec)
        # Environment values (particularly credentials) are never written to disk.
        manifest = {k: v for k, v in job.items() if k != "env"}
        manifest.update(input_artifacts=records, yaml_inputs=materialized, env_names=sorted(job["env"]))
        json_write(directory / "resolved.json", manifest)
        environment = os.environ.copy()
        for name, value in job["env"].items():
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = str(value)
        environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        # Upstream LLM-FE accepts both names but later requires API_KEY itself.
        if job["args"].get("--use_api") and not environment.get("API_KEY") and environment.get("OPENAI_API_KEY"):
            environment["API_KEY"] = environment["OPENAI_API_KEY"]
        command = [str(job["python"]), "-u", str(ENTRY), str(spec_path)]
        if cancel is not None and cancel.is_set():
            raise LaunchError("Launch was cancelled before starting the process.")
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            process = subprocess.Popen(command, cwd=job["cwd"], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, **kwargs)
            timeout = job.get("wall_timeout_seconds")
            deadline = time.monotonic() + timeout if timeout else None
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    raise LaunchError("Launch was cancelled.")
                if deadline is not None and time.monotonic() >= deadline:
                    raise LaunchError(f"Job exceeded wall_timeout_seconds={timeout}")
                time.sleep(0.05)
            code = process.returncode
        record["returncode"] = code
        if code != 0:
            raise LaunchError(f"Process exited with code {code}; see {directory / 'stderr.log'}")
        record["artifacts"] = collect_outputs(job)
        json_write(directory / "artifacts.json", {"version": 1, "job_id": job["id"], "artifacts": record["artifacts"]})
        record["status"] = "completed"
    except BaseException as exc:
        if process is not None:
            stop_process(process)
        record.update(status="failed", error=str(exc))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        json_write(directory / "status.json", record)
    return record


def execute_jobs(jobs: dict, workers: int, run_dir: Path) -> dict:
    pending, running, results = dict(jobs), {}, {}
    cancel = threading.Event()
    def write_run_status():
        active = set(running.values())
        states = {name: results.get(name, {"id": name, "status": "running" if name in active else "pending", "artifacts": {}})
                  for name in jobs}
        json_write(run_dir / "run.json", {"version": 1, "jobs": states})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            while pending or running:
                changed = False
                for name, job in list(pending.items()):
                    dependencies = job["depends_on"]
                    if any(parent in results and results[parent]["status"] != "completed" for parent in dependencies):
                        results[name] = {"id": name, "status": "blocked", "error": "A required job failed.", "artifacts": {}}
                        json_write(Path(job["job_dir"]) / "status.json", results[name])
                        del pending[name]
                        changed = True
                    elif all(parent in results for parent in dependencies) and len(running) < workers:
                        try:
                            resolved = resolve_references(job, results)
                        except LaunchError as exc:
                            results[name] = {"id": name, "status": "failed", "error": str(exc), "artifacts": {}}
                            json_write(Path(job["job_dir"]) / "status.json", results[name])
                        else:
                            future = executor.submit(run_job, resolved, cancel)
                            running[future] = name
                            print(f"[start] {name}", flush=True)
                        del pending[name]
                        changed = True
                if changed:
                    write_run_status()
                if not running:
                    if pending:
                        # A failed reference may have unblocked a dependent failure.
                        continue
                    break
                finished, _ = wait(running, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in finished:
                    name = running.pop(future)
                    try:
                        results[name] = future.result()
                    except Exception as exc:
                        results[name] = {"id": name, "status": "failed", "error": str(exc), "artifacts": {}}
                        json_write(Path(jobs[name]["job_dir"]) / "status.json", results[name])
                    print(f"[{results[name]['status']}] {name}", flush=True)
                if finished:
                    write_run_status()
        except BaseException:
            cancel.set()
            # Workers observe cancellation and stop their own process trees.
            for future in running:
                future.cancel()
            for future, name in running.items():
                try:
                    results[name] = future.result()
                except BaseException as exc:
                    results[name] = {"id": name, "status": "failed", "error": str(exc) or "Launch was cancelled.", "artifacts": {}}
                    json_write(Path(jobs[name]["job_dir"]) / "status.json", results[name])
            raise
        finally:
            if cancel.is_set():
                for name in pending:
                    results[name] = {"id": name, "status": "blocked", "error": "Launch was cancelled.", "artifacts": {}}
            write_run_status()
    return results


def parse_overrides(values: list[str]) -> dict:
    result = {}
    for assignment in values:
        if "=" not in assignment:
            raise LaunchError("--var requires NAME=VALUE")
        name, value = assignment.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        if not name or "." in name:
            raise LaunchError("--var names must be top-level variable names.")
        result[name] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--local", help="Local YAML overrides; machine paths belong here, not credentials.")
    parser.add_argument("--select", action="append", default=[], help="Job-id glob; repeat to select multiple groups.")
    parser.add_argument("--var", action="append", default=[], help="Override a variable: NAME=VALUE (JSON scalar or text).")
    parser.add_argument("--jobs", type=int, default=1, help="Maximum concurrent child processes.")
    parser.add_argument("--output-root", default=str(REPO / "launch_runs"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print the resolved plan; this is also the default.")
    mode.add_argument("--run", action="store_true", help="Execute selected jobs.")
    mode.add_argument("--list", action="store_true", help="List expanded job IDs.")
    args = parser.parse_args(argv)
    try:
        if args.jobs < 1:
            raise LaunchError("--jobs must be at least 1")
        config_path = absolute(args.config)
        config = read_yaml(config_path)
        local = read_yaml(absolute(args.local)) if args.local else None
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        run_dir = absolute(args.output_root) / run_id
        jobs = select_jobs(build_jobs(config, run_dir, local, parse_overrides(args.var)), args.select)
        if args.list:
            for name, job in jobs.items():
                print(f"{name}\t{job.get('stage', '')}")
            return 0
        plans = [plan_job(job) for job in jobs.values()]
        if not args.run:
            print(json.dumps({"version": 1, "mode": "plan", "run_dir": str(run_dir), "jobs": plans}, indent=2))
            return 0
        # Validate every non-dependent input before submitting the first process.
        for job in jobs.values():
            if not job["depends_on"]:
                preflight(job, True)
        run_dir.mkdir(parents=True, exist_ok=False)
        input_record(config_path, run_dir / "config.yaml")
        if args.local:
            # The local file is not copied: it can contain private environment values.
            json_write(run_dir / "local_provenance.json", {"path": str(absolute(args.local)), "sha256": sha256(absolute(args.local))})
        json_write(run_dir / "plan.json", {"version": 1, "jobs": plans})
        results = execute_jobs(jobs, args.jobs, run_dir)
        print(f"Run records: {run_dir}")
        return 0 if all(r["status"] == "completed" for r in results.values()) else 1
    except (LaunchError, OSError, SyntaxError) as exc:
        print(f"Launch error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Launch cancelled; child processes stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
