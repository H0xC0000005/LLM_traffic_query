"""Launcher contract tests using only temporary, synthetic child scripts.

No experiment modules, datasets, services or simulator processes are used.
Run with: python -m unittest discover -s launchers/tests -v
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import threading
import unittest


REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from launchers import run


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="launcher-fixtures-")
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def config(self, jobs, **extra):
        return {
            "version": 1,
            "defaults": {"python": sys.executable, "cwd": str(self.root)},
            "jobs": jobs,
            **extra,
        }

    def invoke(self, config, *arguments):
        config_path = self.root / "fixture-config.yaml"
        # JSON is a YAML subset supported without any third-party dependencies.
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output_root = self.root / "runs"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = run.main([
                "--config", str(config_path), "--output-root", str(output_root),
                *arguments,
            ])
        return code, stdout.getvalue(), stderr.getvalue(), output_root

    def completed_records(self, output_root):
        directories = list(output_root.iterdir())
        self.assertEqual(len(directories), 1)
        run_dir = directories[0]
        return run_dir, json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["jobs"]

    def test_matrix_expands_nested_values_with_types_and_deep_defaults(self):
        config = self.config(
            [{
                "id": "energy-${encoder}",
                "script": "unused-fixture.py",
                "matrix": {"encoder": ["base", "expert"]},
                "args": {
                    "--enabled": "${encoders.${encoder}.enabled}",
                    "--count": "${encoders.${encoder}.count}",
                    "--output": "${job_dir}/result.json",
                },
                "env": {"JOB": "${encoder}"},
            }],
            variables={"encoders": {
                "base": {"enabled": False, "count": 2},
                "expert": {"enabled": True, "count": 5},
            }},
            defaults={"python": sys.executable, "cwd": str(self.root),
                      "env": {"SHARED": "kept"}, "args": {"--shared": 9}},
        )
        jobs = run.build_jobs(config, self.root / "uncreated-run")
        self.assertEqual(set(jobs), {"energy-base", "energy-expert"})
        self.assertIs(jobs["energy-base"]["args"]["--enabled"], False)
        self.assertIs(jobs["energy-expert"]["args"]["--enabled"], True)
        self.assertIsInstance(jobs["energy-expert"]["args"]["--count"], int)
        self.assertEqual(jobs["energy-expert"]["env"], {"SHARED": "kept", "JOB": "expert"})
        self.assertEqual(jobs["energy-expert"]["args"]["--shared"], 9)
        self.assertNotIn("--enabled", run.flag_args(jobs["energy-base"]["args"]))
        self.assertIn("--enabled", run.flag_args(jobs["energy-expert"]["args"]))
        self.assertFalse((self.root / "uncreated-run").exists())

    def test_selection_includes_transitive_artifact_dependencies(self):
        jobs = run.build_jobs(self.config([
            {"id": "features", "script": "unused.py",
             "outputs": {"source": "${job_dir}/features.py"}},
            {"id": "train", "script": "unused.py", "inputs": [
                {"from_job": "features", "artifact": "source"}],
             "outputs": {"model": "${job_dir}/model.json"}},
            {"id": "evaluate", "script": "unused.py", "args": {
                "--model": {"from_job": "train", "artifact": "model"}}},
            {"id": "unrelated", "script": "unused.py"},
        ]), self.root / "run")
        selected = run.select_jobs(jobs, ["eval*"])
        self.assertEqual(set(selected), {"features", "train", "evaluate"})
        self.assertEqual(selected["evaluate"]["depends_on"], ["train"])
        with self.assertRaisesRegex(run.LaunchError, "matches no jobs"):
            run.select_jobs(jobs, ["missing*"])

    def test_unknown_artifact_name_is_rejected_before_any_launch(self):
        script = self.write("no-op.py", "print('unused fixture')\n")
        config = self.config([
            {"id": "producer", "script": str(script),
             "outputs": {"known": "${job_dir}/known.json"}},
            {"id": "consumer", "script": str(script), "args": {
                "--input": {"from_job": "producer", "artifact": "misspelled"}}},
        ])
        with self.assertRaisesRegex(run.LaunchError, "undeclared artifact producer:misspelled"):
            run.build_jobs(config, self.root / "uncreated-run")
        code, _, stderr, output_root = self.invoke(config, "--plan")
        self.assertEqual(code, 2)
        self.assertIn("undeclared artifact producer:misspelled", stderr)
        self.assertFalse(output_root.exists())

    def test_legacy_flag_is_rejected_even_when_disabled(self):
        script = self.write("legacy.py", """
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--use_expert_features', action='store_true')
            parser.add_argument('--use-expert-features', action='store_true')
        """)
        for flag in ("--use_expert_features", "--use-expert-features"):
            for value in (True, False, None):
                with self.subTest(flag=flag, value=value):
                    with self.assertRaisesRegex(run.LaunchError, "Legacy flag"):
                        run.flag_args({flag: value})
                    jobs = run.build_jobs(self.config([
                        {"id": "legacy", "script": str(script), "args": {flag: value}},
                    ]), self.root / "run")
                    with self.assertRaisesRegex(run.LaunchError, "Legacy flag"):
                        run.preflight(jobs["legacy"], False)

    def test_default_plan_creates_nothing_and_never_imports_sources(self):
        script_marker = self.root / "script-was-executed"
        extractor_marker = self.root / "extractor-was-imported"
        script = self.write("do-not-run.py", f"""
            import argparse
            from pathlib import Path
            Path({str(script_marker)!r}).write_text('executed')
            parser = argparse.ArgumentParser()
            parser.add_argument('--template')
            raise RuntimeError('The plan must never execute this fixture')
        """)
        extractor = self.write("do-not-import.py", f"""
            from pathlib import Path
            Path({str(extractor_marker)!r}).write_text('imported')
            def extract_expert_features():
                return 'unused'
            raise RuntimeError('The plan must never import this fixture')
        """)
        template = self.write("original-template.yaml", '{"setting": {"before": 1}}')
        config = self.config([{
            "id": "plan-only", "script": str(script),
            "args": {"--template": "${yaml_inputs.template}"},
            "extractor": {"path": str(extractor), "symbol": "extract_expert_features"},
            "yaml_inputs": {"template": {
                "source": str(template), "overrides": {"setting": {"after": 2}}}},
            "inputs": ["${yaml_inputs.template}"],
        }])
        config_path = self.root / "fixture-config.yaml"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        before = {str(p.relative_to(self.root)): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        code, stdout, stderr, output_root = self.invoke(config)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["mode"], "plan")
        self.assertFalse(output_root.exists())
        self.assertFalse(script_marker.exists())
        self.assertFalse(extractor_marker.exists())
        after = {str(p.relative_to(self.root)): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_child_preserves_arguments_environment_cwd_and_logs(self):
        script = self.write("record-child.py", """
            import argparse
            import json
            import os
            from pathlib import Path
            import sys
            p = argparse.ArgumentParser()
            p.add_argument('--text')
            p.add_argument('--enabled', action='store_true')
            p.add_argument('--output')
            args = p.parse_args()
            Path(args.output).write_text(json.dumps({
                'text': args.text, 'enabled': args.enabled, 'cwd': str(Path.cwd()),
                'value': os.environ.get('FIXTURE_VALUE'),
                'argv0': sys.argv[0], 'stdin': sys.stdin.read(),
            }), encoding='utf-8')
            print('fixture stdout', flush=True)
            print('fixture stderr', file=sys.stderr, flush=True)
        """)
        literal = 'literal spaces & $not_a_shell_command'
        code, _, stderr, output_root = self.invoke(self.config([{
            "id": "record", "script": str(script),
            "env": {"FIXTURE_VALUE": "fixture-private-value"},
            "args": {"--text": literal, "--enabled": True,
                     "--output": "${job_dir}/observed.json"},
            "outputs": {"observed": {"path": "${job_dir}/observed.json"}},
        }]), "--run")
        self.assertEqual(code, 0, stderr)
        run_dir, records = self.completed_records(output_root)
        record = records["record"]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["returncode"], 0)
        observed = json.loads(Path(record["artifacts"]["observed"]).read_text(encoding="utf-8"))
        self.assertEqual(observed["text"], literal)
        self.assertIs(observed["enabled"], True)
        self.assertEqual(Path(observed["cwd"]), self.root)
        self.assertEqual(observed["value"], "fixture-private-value")
        self.assertEqual(Path(observed["argv0"]), script)
        self.assertEqual(observed["stdin"], "")
        job_dir = run_dir / "jobs" / "record"
        self.assertIn("fixture stdout", (job_dir / "stdout.log").read_text())
        self.assertIn("fixture stderr", (job_dir / "stderr.log").read_text())
        self.assertNotIn("fixture-private-value", (job_dir / "resolved.json").read_text())
        index = json.loads((job_dir / "artifacts.json").read_text())
        self.assertEqual(index["artifacts"]["observed"], record["artifacts"]["observed"])

    def test_parallel_extractors_are_isolated_and_imported_in_script_order(self):
        script = self.write("unchanged-consumer.py", """
            import argparse
            import json
            import os
            from pathlib import Path
            os.environ['FIXTURE_IMPORT_READY'] = 'ready'
            from expert_feature_extractor import extract_expert_features
            p = argparse.ArgumentParser()
            p.add_argument('--output')
            args = p.parse_args()
            Path(args.output).write_text(json.dumps(extract_expert_features()), encoding='utf-8')
        """)
        # A normal canonical module must not be edited or imported by either child.
        canonical = self.write("expert_feature_extractor.py", "raise RuntimeError('wrong source')\n")
        jobs, sources = [], {}
        for name in ("left", "right"):
            source = self.write(f"selected-{name}.py", f"""
                import os
                assert os.environ.get('FIXTURE_IMPORT_READY') == 'ready', 'imported too early'
                def extract_expert_features():
                    return {{'selection': {name!r}, 'source': __file__}}
            """)
            sources[name] = source.read_bytes()
            jobs.append({
                "id": name, "script": str(script),
                "extractor": {"path": str(source), "symbol": "extract_expert_features"},
                "args": {"--output": "${job_dir}/observed.json"},
                "outputs": {"observed": {"path": "${job_dir}/observed.json"}},
            })
        original_script, original_canonical = script.read_bytes(), canonical.read_bytes()
        code, _, stderr, output_root = self.invoke(self.config(jobs), "--run", "--jobs", "2")
        self.assertEqual(code, 0, stderr)
        run_dir, records = self.completed_records(output_root)
        for name in ("left", "right"):
            record = records[name]
            self.assertEqual(record["status"], "completed")
            observed = json.loads(Path(record["artifacts"]["observed"]).read_text())
            self.assertEqual(observed["selection"], name)
            snapshot = run_dir / "jobs" / name / "inputs" / "extractor.py"
            self.assertEqual(Path(observed["source"]), snapshot)
            self.assertEqual(snapshot.read_bytes(), sources[name])
            self.assertEqual((self.root / f"selected-{name}.py").read_bytes(), sources[name])
        self.assertEqual(script.read_bytes(), original_script)
        self.assertEqual(canonical.read_bytes(), original_canonical)
        self.assertEqual(list(self.root.glob("**/__pycache__")), [])

    def test_declared_artifact_flows_from_train_to_selected_evaluation(self):
        train = self.write("fixture-train.py", """
            import argparse
            import json
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument('--out')
            args = p.parse_args()
            directory = Path(args.out)
            directory.mkdir()
            (directory / 'checkpoint.json').write_text(json.dumps({'learned': 17}))
        """)
        evaluate = self.write("fixture-evaluate.py", """
            import argparse
            import json
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument('--checkpoint')
            p.add_argument('--output')
            args = p.parse_args()
            model = json.loads(Path(args.checkpoint).read_text())
            Path(args.output).write_text(json.dumps({'verified': model['learned']}))
        """)
        code, _, stderr, output_root = self.invoke(self.config([
            {"id": "train", "script": str(train),
             "args": {"--out": "${job_dir}/models"},
             "outputs": {"checkpoint": {"glob": "${job_dir}/models/*.json", "one": True}}},
            {"id": "evaluate", "script": str(evaluate),
             "args": {"--checkpoint": {"from_job": "train", "artifact": "checkpoint"},
                      "--output": "${job_dir}/evaluation.json"},
             "outputs": {"metrics": "${job_dir}/evaluation.json"}},
            {"id": "unrelated", "script": str(self.root / "does-not-exist.py")},
        ]), "--run", "--select", "evaluate", "--jobs", "2")
        self.assertEqual(code, 0, stderr)
        _, records = self.completed_records(output_root)
        self.assertEqual(set(records), {"train", "evaluate"})
        self.assertEqual(records["train"]["status"], "completed")
        self.assertEqual(records["evaluate"]["status"], "completed")
        metrics = Path(records["evaluate"]["artifacts"]["metrics"])
        self.assertEqual(json.loads(metrics.read_text()), {"verified": 17})

    def test_missing_optional_artifact_fails_only_its_consumer(self):
        producer = self.write("optional-producer.py", "print('no optional file produced')\n")
        unexpected_marker = self.root / "consumer-ran"
        consumer = self.write("optional-consumer.py", f"""
            import argparse
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument('--input')
            p.parse_args()
            Path({str(unexpected_marker)!r}).write_text('unexpected')
        """)
        unrelated = self.write("independent-child.py", """
            import argparse
            from pathlib import Path
            import time
            p = argparse.ArgumentParser()
            p.add_argument('--output')
            args = p.parse_args()
            time.sleep(0.1)
            Path(args.output).write_text('independent child finished')
        """)
        code, _, _, output_root = self.invoke(self.config([
            {"id": "producer", "script": str(producer), "outputs": {
                "optional": {"path": "${job_dir}/not-produced.txt", "required": False}}},
            {"id": "consumer", "script": str(consumer), "args": {
                "--input": {"from_job": "producer", "artifact": "optional"}}},
            {"id": "independent", "script": str(unrelated),
             "args": {"--output": "${job_dir}/finished.txt"},
             "outputs": {"finished": "${job_dir}/finished.txt"}},
        ]), "--run", "--jobs", "2")
        self.assertEqual(code, 1)
        run_dir, records = self.completed_records(output_root)
        self.assertEqual(records["producer"]["status"], "completed")
        self.assertIsNone(records["producer"]["artifacts"]["optional"])
        self.assertEqual(records["consumer"]["status"], "failed")
        self.assertIn("error", records["consumer"])
        self.assertFalse(unexpected_marker.exists())
        self.assertFalse((run_dir / "jobs" / "consumer" / "stdout.log").exists())
        self.assertEqual(records["independent"]["status"], "completed")
        self.assertEqual(Path(records["independent"]["artifacts"]["finished"]).read_text(),
                         "independent child finished")

    def test_yaml_overrides_materialize_per_run_without_changing_original(self):
        original = {
            "settings": {"keep": 1, "value": "original", "list": [1, 2]},
            "original": True,
        }
        template = self.write("source-template.yaml", json.dumps(original))
        original_bytes = template.read_bytes()
        script = self.write("yaml-consumer.py", """
            import argparse
            import json
            from pathlib import Path
            p = argparse.ArgumentParser()
            p.add_argument('--template')
            p.add_argument('--output')
            args = p.parse_args()
            # The launcher writes JSON, which is also valid YAML.
            merged = json.loads(Path(args.template).read_text(encoding='utf-8'))
            Path(args.output).write_text(json.dumps({
                'template': args.template, 'merged': merged,
            }), encoding='utf-8')
        """)
        code, _, stderr, output_root = self.invoke(self.config([{
            "id": "materialize", "script": str(script),
            "yaml_inputs": {"template": {
                "source": str(template),
                "overrides": {"settings": {"value": "overridden", "added": 7}},
            }},
            "inputs": ["${yaml_inputs.template}"],
            "args": {"--template": "${yaml_inputs.template}",
                     "--output": "${job_dir}/observed.json"},
            "outputs": {"observed": "${job_dir}/observed.json"},
        }]), "--run")
        self.assertEqual(code, 0, stderr)
        run_dir, records = self.completed_records(output_root)
        self.assertEqual(records["materialize"]["status"], "completed")
        observed = json.loads(Path(records["materialize"]["artifacts"]["observed"]).read_text())
        expected = {
            "settings": {"keep": 1, "value": "overridden", "list": [1, 2], "added": 7},
            "original": True,
        }
        self.assertEqual(observed["merged"], expected)
        materialized = run_dir / "jobs" / "materialize" / "inputs" / "template.yaml"
        self.assertEqual(Path(observed["template"]), materialized)
        self.assertEqual(json.loads(materialized.read_text()), expected)
        self.assertEqual(template.read_bytes(), original_bytes)
        provenance = json.loads((run_dir / "jobs" / "materialize" / "resolved.json").read_text())
        self.assertEqual(Path(provenance["yaml_inputs"]["template"]["source"]), template)

    def test_failed_child_records_exit_code_and_blocks_dependents(self):
        failure = self.write("failure.py", """
            import sys
            print('intentional fixture failure', file=sys.stderr)
            raise SystemExit(7)
        """)
        marker = self.root / "dependent-ran"
        dependent = self.write("dependent.py", f"""
            from pathlib import Path
            Path({str(marker)!r}).write_text('unexpected')
        """)
        code, _, _, output_root = self.invoke(self.config([
            {"id": "failure", "script": str(failure)},
            {"id": "dependent", "script": str(dependent), "depends_on": ["failure"]},
        ]), "--run", "--jobs", "2")
        self.assertEqual(code, 1)
        run_dir, records = self.completed_records(output_root)
        self.assertEqual(records["failure"]["status"], "failed")
        self.assertEqual(records["failure"]["returncode"], 7)
        self.assertEqual(records["dependent"]["status"], "blocked")
        self.assertFalse(marker.exists())
        self.assertFalse((run_dir / "jobs" / "dependent" / "stdout.log").exists())
        self.assertIn("intentional fixture failure", (run_dir / "jobs" / "failure" / "stderr.log").read_text())

    def test_timeout_terminates_synthetic_child_and_records_failure(self):
        script = self.write("sleep-fixture.py", """
            import time
            print('fixture is waiting', flush=True)
            time.sleep(10)
        """)
        code, _, _, output_root = self.invoke(self.config([{
            "id": "timeout", "script": str(script), "wall_timeout_seconds": 0.5,
        }]), "--run")
        self.assertEqual(code, 1)
        run_dir, records = self.completed_records(output_root)
        self.assertEqual(records["timeout"]["status"], "failed")
        self.assertIn("wall_timeout_seconds", records["timeout"]["error"])
        self.assertIn("finished_at", records["timeout"])

    def test_expert_evaluation_metadata_requires_explicit_extractor(self):
        script = self.write("fixture-evaluation-preflight.py", """
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument('--checkpoint')
            p.parse_args()
        """)
        checkpoint = self.write("fixture-checkpoint.pt", "synthetic checkpoint; never loaded")
        metadata = self.write("fixture-checkpoint.json", json.dumps({"addon_encoder_name": "expert"}))
        config = self.config([{
            "id": "evaluation", "stage": "tsc_eval", "script": str(script),
            "extractor": None,
            "args": {"--checkpoint": str(checkpoint)},
            "inputs": [str(checkpoint)],
        }])
        job = run.build_jobs(config, self.root / "uncreated-run")["evaluation"]
        warnings = run.plan_job(job)["warnings"]
        self.assertTrue(any("explicit matching extractor" in warning for warning in warnings))
        with self.assertRaisesRegex(run.LaunchError, "explicit matching extractor"):
            run.preflight(job, True)
        self.assertIn(str(metadata), run.configured_inputs(job))

        extractor = self.write("compatible-fixture-extractor.py", """
            def extract_expert_features():
                return 'not imported during preflight'
        """)
        job["extractor"] = {"path": str(extractor), "symbol": "extract_expert_features"}
        self.assertEqual(run.plan_job(job)["warnings"], [])
        self.assertEqual(run.preflight(job, True), [])
        self.assertIn(str(metadata), run.configured_inputs(job))
        self.assertFalse((self.root / "uncreated-run").exists())

    def test_pre_cancelled_job_records_failure_without_starting_child(self):
        marker = self.root / "cancelled-child-ran"
        script = self.write("cancelled-child.py", f"""
            from pathlib import Path
            Path({str(marker)!r}).write_text('unexpected execution')
        """)
        job = run.build_jobs(self.config([{
            "id": "cancelled", "script": str(script),
        }]), self.root / "run")["cancelled"]
        cancelled = threading.Event()
        cancelled.set()
        record = run.run_job(job, cancelled)
        self.assertEqual(record["status"], "failed")
        self.assertIn("cancelled before starting", record["error"])
        self.assertIn("finished_at", record)
        self.assertFalse(marker.exists())
        job_dir = Path(job["job_dir"])
        self.assertFalse((job_dir / "stdout.log").exists())
        persisted = json.loads((job_dir / "status.json").read_text())
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["error"], record["error"])

    def test_sumo_configuration_discovers_relative_input_files_without_simulation(self):
        script = self.write("no-simulator.py", "raise RuntimeError('must not execute')\n")
        network = self.write("scenario/network.net.xml", "<net/>\n")
        routes_a = self.write("scenario/routes/a.rou.xml", "<routes/>\n")
        routes_b = self.write("scenario/routes/b.rou.xml", "<routes/>\n")
        additional = self.write("scenario/extra.add.xml", "<additional/>\n")
        sumocfg = self.write("scenario/fixture.sumocfg", """
            <configuration>
              <input>
                <net-file value="network.net.xml"/>
                <route-files value="routes/a.rou.xml, routes/b.rou.xml"/>
                <additional-files value="extra.add.xml"/>
              </input>
            </configuration>
        """)
        job = run.build_jobs(self.config([{
            "id": "sumo-inputs", "script": str(script), "inputs": [str(sumocfg)],
        }]), self.root / "uncreated-run")["sumo-inputs"]
        self.assertEqual(set(run.configured_inputs(job)),
                         {str(path) for path in (sumocfg, network, routes_a, routes_b, additional)})
        self.assertEqual(run.preflight(job, True), [])
        self.assertFalse((self.root / "uncreated-run").exists())


if __name__ == "__main__":
    unittest.main()
