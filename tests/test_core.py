import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from agent_gym.contracts import TaskSpec
from agent_gym.episode import Episode
from agent_gym.export import export_run
from agent_gym.reference import EmptyAgent, ReferenceAgent
from agent_gym.registry import make, task_names
from agent_gym.runner import run_batch, run_episode
from agent_gym.service import GymService, serve_jsonl
from agent_gym.storage import RunLease
from agent_gym.tasks.optics import reflectance, oracle_reflectance


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gym tests ")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_reference_and_negative_controls(self):
        for family in task_names():
            for difficulty in (1, 2, 3):
                for seed in range(4):
                    spec = TaskSpec(family, seed, difficulty=difficulty)
                    with self.subTest(task=spec.id):
                        result = run_episode(spec, self.root, lambda _: ReferenceAgent(), agent_name="reference")
                        self.assertEqual(result["status"], "passed", result)
                        self.assertEqual(result["reward"], 1)
                result = run_episode(spec, self.root, lambda _: EmptyAgent(), agent_name="empty")
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["reward"], 0)

    def test_reproducible_worlds_and_distinct_splits(self):
        for family in task_names():
            first, second = make(TaskSpec(family)), make(TaskSpec(family))
            self.assertEqual(first.observation(), second.observation())
            tool = first.environment.tools[0].name
            self.assertEqual(first.environment.call(tool, {}), second.environment.call(tool, {}))
        a = make(TaskSpec("rc-identification"))
        b = make(TaskSpec("rc-identification"))
        c = make(TaskSpec("rc-identification", split="test"))
        args = {"frequency_hz": 1000}
        self.assertEqual(a.environment.call("measure_response", args), b.environment.call("measure_response", args))
        self.assertNotEqual(a.environment.call("measure_response", args)["real"], c.environment.call("measure_response", args)["real"])

    def test_independent_physics_expressions_and_analytic_solution(self):
        for n in (1.2, 1.7, 2.3):
            for substrate in (1.5, 2.2):
                for thickness in (20, 137, 400):
                    a = reflectance(n, substrate, thickness, 550)
                    b = oracle_reflectance(n, substrate, thickness, 550)
                    self.assertAlmostEqual(a, b, places=13)
                    self.assertTrue(0 <= a <= 1)
        n = math.sqrt(1.7)
        self.assertLess(oracle_reflectance(n, 1.7, 550 / (4 * n), 550), 1e-28)

    def test_verifier_does_not_read_mutated_instrument(self):
        task = make(TaskSpec("rc-identification"))
        truth = {"gain": task.environment.gain, "cutoff_hz": task.environment.cutoff}
        task.environment.gain *= 2
        self.assertTrue(task.verify(truth).passed)
        self.assertFalse(task.verify({**truth, "gain": truth["gain"] * 2}).passed)

    def test_observations_are_not_mutable_world_references(self):
        with Episode(make(TaskSpec("thin-film-design")), self.root) as episode:
            first = episode.step({"tool": "design_conditions", "arguments": {}})
            first["result"]["materials"].clear()
            second = episode.step({"tool": "design_conditions", "arguments": {}})
            self.assertTrue(second["result"]["materials"])

    def test_correctable_actions_then_limit(self):
        task = make(TaskSpec("rc-identification"))
        task.max_actions = 2
        with Episode(task, self.root) as episode:
            response = episode.step({"tool": "missing", "arguments": {}})
            self.assertEqual(response["type"], "tool_error")
            response = episode.step({"tool": "measure_response", "arguments": {"frequency_hz": -1}})
            self.assertEqual(response["result"]["status"], "truncated")
            self.assertIsNone(response["result"]["reward"])

    def test_final_submission_uses_last_action(self):
        task = make(TaskSpec("rc-identification"))
        task.max_actions = 1
        with Episode(task, self.root) as episode:
            self.assertEqual(episode.step({"submit": {}})["result"]["status"], "failed")

    def test_errors_and_cleanup_are_not_negative_rewards(self):
        task = make(TaskSpec("rc-identification"))
        task.verify = lambda _: 1 / 0
        with Episode(task, self.root) as episode:
            result = episode.step({"submit": {}})["result"]
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["reward"])
        class BadCleanup(ReferenceAgent):
            def close(self):
                raise RuntimeError("cleanup example")
        result = run_episode(TaskSpec("rc-identification"), self.root, lambda _: BadCleanup(), agent_name="broken")
        persisted = json.loads((Path(result["attempt_path"]) / "result.json").read_text())
        self.assertEqual(persisted["status"], "error")
        self.assertIsNone(persisted["reward"])
        self.assertIn("cleanup example", persisted["cleanup_error"])

    def test_invalid_environment_result_is_infrastructure_error(self):
        task = make(TaskSpec("rc-identification"))
        task.environment.call = lambda *_: {"bad": math.nan}
        with Episode(task, self.root) as episode:
            result = episode.step({"tool": "instrument_info", "arguments": {}})["result"]
            self.assertEqual(result["status"], "error")

    def test_parallel_attempts_resume_and_export(self):
        specs = [TaskSpec(f, seed) for f in task_names() for seed in range(3)]
        first = run_batch(specs, self.root, lambda _: ReferenceAgent(), jobs=4)
        self.assertEqual(first["counts"]["passed"], 9)
        self.assertEqual(len({r["episode_id"] for r in first["results"]}), 9)
        def forbidden(_):
            raise AssertionError("Completed work must not restart")
        second = run_batch(specs, self.root, forbidden, jobs=2, resume=True)
        self.assertEqual(first["results"], second["results"])
        target = self.root / "dataset.jsonl"
        self.assertEqual(export_run(self.root, target), 9)
        data = [json.loads(line) for line in target.read_text().splitlines()]
        self.assertEqual(data[0]["trajectory"][0]["kind"], "observation")
        self.assertEqual(data[0]["trajectory"][-1]["kind"], "action")
        with self.assertRaises(FileExistsError):
            export_run(self.root, target)

    def test_retry_retains_failed_attempt(self):
        spec = TaskSpec("rc-identification")
        def broken(_):
            raise RuntimeError("offline")
        first = run_batch([spec], self.root, broken)
        self.assertEqual(first["counts"]["error"], 1)
        second = run_batch([spec], self.root, lambda _: ReferenceAgent(), resume=True, retry_errors=True)
        self.assertEqual(second["counts"]["passed"], 1)
        self.assertEqual(len(list((self.root / spec.id).glob("*/result.json"))), 2)

    def test_output_lease(self):
        with RunLease(self.root):
            with self.assertRaises(RuntimeError):
                with RunLease(self.root):
                    pass

    def test_limits_reject_nonfinite_values(self):
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                Episode(make(TaskSpec("rc-identification")), self.root, seconds=value)

    def test_service_owns_separate_environments(self):
        service = GymService(self.root, max_active=2)
        self.addCleanup(service.close_all)
        one = service.open("rc-identification")
        two = service.open("rc-identification")
        self.assertNotEqual(one["episode_id"], two["episode_id"])
        action = {"tool": "measure_response", "arguments": {"frequency_hz": 300}}
        a = service.action(one["episode_id"], action)
        b = service.action(two["episode_id"], action)
        self.assertEqual(a["result"], b["result"])
        with self.assertRaises(ValueError):
            service.open("thin-film-design")
        service.close(one["episode_id"])
        service.open("thin-film-design")

    def test_jsonl_service(self):
        output = io.StringIO()
        serve_jsonl(GymService(self.root), io.StringIO('{"id":1,"method":"tasks"}\n{"id":2,"method":"open","params":{"family":"rc-identification"}}\n'), output)
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[0]["id"], 1)
        self.assertEqual(messages[1]["result"]["type"], "start")
        saved = list(self.root.glob("*/*/result.json"))
        self.assertEqual(json.loads(saved[0].read_text())["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
