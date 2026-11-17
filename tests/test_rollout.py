from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest

from agent_gym.contracts import Task, TaskSpec, Tool, Verdict
from agent_gym.rollout import PolicyVersion, RolloutBatch, export_rollouts
from agent_gym.suite import TaskSuite


class CounterEnvironment:
    tools = (Tool("increment", "Increment and observe a counter", {"type": "object"}),)

    def __init__(self, *, barrier=None, failure=False, bad_cleanup=False):
        self.count = 0
        self.closed = False
        self.barrier, self.failure, self.bad_cleanup = barrier, failure, bad_cleanup

    def call(self, name, arguments):
        if self.barrier:
            self.barrier.wait(timeout=3)
        if self.failure:
            raise RuntimeError("measurement unavailable")
        self.count += 1
        return {"count": self.count}

    def close(self):
        self.closed = True
        if self.bad_cleanup:
            raise RuntimeError("cleanup unavailable")


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.specs = [TaskSpec("rc-identification", seed=seed) for seed in range(7)]
        self.suite = TaskSuite(self.specs, attempts_per_task=4, name="repeated-tasks", metadata={"source_revision": "1"})

    def test_content_identity_and_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            self.suite.save(path)
            loaded = TaskSuite.load(path)
            self.assertEqual(loaded.to_dict(), self.suite.to_dict())
            data = loaded.to_dict()
            data["metadata"]["source_revision"] = "2"
            with self.assertRaisesRegex(ValueError, "suite_id"):
                TaskSuite.from_dict(data)
            different = TaskSuite(self.specs, attempts_per_task=4, metadata={"source_revision": "2"})
            self.assertNotEqual(different.suite_id, self.suite.suite_id)

    def test_group_sharding_is_disjoint_and_complete(self):
        global_tickets = list(self.suite.sampler(seed=23, epoch=4))
        shards = [list(self.suite.sampler(seed=23, epoch=4, rank=rank, world_size=3)) for rank in range(3)]
        self.assertEqual({t.rollout_id for t in global_tickets}, {t.rollout_id for shard in shards for t in shard})
        owners = {}
        for rank, shard in enumerate(shards):
            for ticket in shard:
                self.assertEqual(owners.setdefault(ticket.group_id, rank), rank)
        self.assertEqual(len(owners), 7)
        group = global_tickets[:4]
        self.assertEqual(len({t.group_id for t in group}), 1)
        self.assertEqual(len({t.spec for t in group}), 1)
        self.assertEqual(len({t.sample_seed for t in group}), 4)

    def test_exact_sampling_resume_and_configuration_binding(self):
        sampler = self.suite.sampler(seed=4, rank=1, world_size=2)
        first = sampler.take(3)
        checkpoint = sampler.state_dict()
        expected = list(sampler)
        resumed = self.suite.sampler(seed=4, rank=1, world_size=2)
        resumed.load_state_dict(checkpoint)
        self.assertEqual(list(resumed), expected)
        self.assertEqual(len(first) + len(expected), sampler.total)
        for kwargs in ({"seed": 5, "rank": 1, "world_size": 2}, {"seed": 4, "rank": 0, "world_size": 2},
                       {"seed": 4, "rank": 1, "world_size": 3}, {"seed": 4, "rank": 1, "world_size": 2, "epoch": 1}):
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.suite.sampler(**kwargs).load_state_dict(checkpoint)

    def test_empty_worker_and_unique_tasks(self):
        self.assertEqual(list(self.suite.sampler(rank=8, world_size=9)), [])
        with self.assertRaisesRegex(ValueError, "unique"):
            TaskSuite([self.specs[0], self.specs[0]])


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.envs = []
        self.suite = TaskSuite([TaskSpec("counter-fixture", seed=i) for i in range(3)], attempts_per_task=2)
        self.policy = PolicyVersion("fixture-policy", "checkpoint-10", {"temperature": 0.8})

    def make(self, spec):
        env = CounterEnvironment()
        self.envs.append(env)
        task = Task(spec, "Increment a counter, then submit its value.", env,
                    lambda answer: Verdict(answer == {"count": env.count}, float(answer == {"count": env.count}), "counter"))
        task.data_reference = {"record_id": str(spec.seed), "release_id": "fixture-release"}
        return task

    def batch(self, **kwargs):
        return RolloutBatch(self.root / "run", slots=3, policy=self.policy, task_factory=self.make, **kwargs)

    def test_partial_steps_refill_group_and_policy_metadata(self):
        tickets = self.suite.sampler(shuffle=False).take(3)
        with self.batch() as batch:
            observations = batch.reset({0: tickets[0], 2: tickets[1]})
            self.assertEqual(observations[0]["rollout"]["group_id"], observations[2]["rollout"]["group_id"])
            self.assertNotEqual(observations[0]["rollout"]["episode_id"], observations[2]["rollout"]["episode_id"])
            self.assertEqual(observations[0]["rollout"]["data_reference"]["release_id"], "fixture-release")
            one = batch.step({0: {"tool": "increment", "arguments": {}}},
                             policy_data={0: {"token_ids": [1, 2], "logprobs": [-0.2, -0.3]}})
            self.assertEqual(one[0]["observation"]["result"]["count"], 1)
            self.assertFalse(one[0]["done"])
            self.assertIsNone(one[0]["reward"])
            final = batch.step({0: {"submit": {"count": 1}}, 2: {"submit": {"count": 10}}})
            self.assertTrue(final[0]["terminated"])
            self.assertTrue(final[0]["scored"])
            self.assertEqual(final[0]["reward"], 1)
            self.assertTrue(final[2]["terminated"])
            self.assertEqual(final[2]["reward"], 0)
            batch.reset({2: tickets[2]}, policy=PolicyVersion("fixture-policy", "checkpoint-11"))
            self.assertEqual(batch.active_slots, (2,))
        self.assertTrue(all(env.closed for env in self.envs))
        self.assertEqual(batch.summary()["counts"], {"passed": 1, "failed": 1, "truncated": 0, "error": 0, "cancelled": 1})
        self.assertEqual(export_rollouts(self.root / "run", self.root / "export.jsonl"), 3)
        exported = [json.loads(line) for line in (self.root / "export.jsonl").read_text().splitlines()]
        self.assertEqual(exported[0]["policy_steps"], [{"step": 1, "data": {"token_ids": [1, 2], "logprobs": [-0.2, -0.3]}}])
        self.assertEqual(exported[1]["policy_steps"], [])
        self.assertEqual(exported[2]["rollout"]["policy"]["revision"], "checkpoint-11")
        self.assertIsNone(exported[2]["reward"])
        self.assertEqual(export_rollouts(self.root / "run", self.root / "scored.jsonl", include_unscored=False), 2)

    def test_batch_steps_execute_slots_concurrently(self):
        barrier = threading.Barrier(2)
        def factory(spec):
            task = self.make(spec)
            task.environment.barrier = barrier
            return task
        with RolloutBatch(self.root / "run", slots=2, workers=2, policy=self.policy, task_factory=factory) as batch:
            batch.reset(dict(enumerate(self.suite.sampler().take(2))))
            response = batch.step({0: {"tool": "increment", "arguments": {}}, 1: {"tool": "increment", "arguments": {}}})
            self.assertEqual([response[i]["observation"]["result"]["count"] for i in range(2)], [1, 1])

    def test_completed_slot_refills_while_other_world_continues(self):
        tickets = self.suite.sampler(shuffle=False).take(3)
        with self.batch() as batch:
            batch.reset({0: tickets[0], 1: tickets[1]})
            batch.step({1: {"tool": "increment", "arguments": {}}})
            batch.step({0: {"submit": {"count": 0}}})
            batch.reset({0: tickets[2]})
            values = batch.step({0: {"tool": "increment", "arguments": {}}, 1: {"tool": "increment", "arguments": {}}})
            self.assertEqual(values[0]["observation"]["result"]["count"], 1)
            self.assertEqual(values[1]["observation"]["result"]["count"], 2)

    def test_simultaneous_callers_preserve_per_slot_order(self):
        with self.batch() as batch:
            batch.reset({0: next(self.suite.sampler())})
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(batch.step, {0: {"tool": "increment", "arguments": {}}}) for _ in range(2)]
                values = [future.result()[0]["observation"]["result"]["count"] for future in futures]
            self.assertEqual(sorted(values), [1, 2])
            batch.step({0: {"submit": {"count": 2}}})
        events = [json.loads(line) for line in next((self.root / "run").glob("*/*/events.jsonl")).read_text().splitlines()]
        self.assertEqual([e["step"] for e in events if e["kind"] == "action"], [1, 2, 3])

    def test_infrastructure_failure_and_truncation_are_unscored(self):
        def factory(spec):
            task = self.make(spec)
            task.max_actions = 1
            task.environment.failure = spec.seed == 0
            return task
        suite = TaskSuite([TaskSpec("counter-fixture", seed=i) for i in range(2)])
        with RolloutBatch(self.root / "run", slots=2, policy=self.policy, task_factory=factory) as batch:
            batch.reset(dict(enumerate(suite.sampler(shuffle=False).take(2))))
            rows = batch.step({i: {"tool": "increment", "arguments": {}} for i in range(2)})
            self.assertEqual(rows[0]["status"], "error")
            self.assertTrue(rows[1]["truncated"])
            for row in rows.values():
                self.assertTrue(row["done"])
                self.assertFalse(row["scored"])
                self.assertIsNone(row["reward"])

    def test_setup_failure_does_not_break_other_slots(self):
        def factory(spec):
            if spec.seed == 0:
                raise RuntimeError("fixture setup failed")
            return self.make(spec)
        suite = TaskSuite([TaskSpec("counter-fixture", seed=i) for i in range(2)])
        with RolloutBatch(self.root / "run", slots=2, policy=self.policy, task_factory=factory) as batch:
            rows = batch.reset(dict(enumerate(suite.sampler(shuffle=False).take(2))))
            self.assertEqual(rows[0]["status"], "error")
            self.assertFalse(rows[1]["done"])
            batch.step({1: {"submit": {"count": 0}}})
        self.assertEqual(export_rollouts(self.root / "run", self.root / "export.jsonl"), 1)
        self.assertEqual(batch.summary()["counts"]["error"], 1)
        saved = json.loads(next((self.root / "run/setup-errors").glob("*.json")).read_text())
        self.assertIsNone(saved["result"]["reward"])

    def test_reset_validation_is_atomic(self):
        tickets = self.suite.sampler().take(3)
        with self.batch() as batch:
            batch.reset({0: tickets[0]})
            with self.assertRaisesRegex(ValueError, "already active"):
                batch.reset({1: tickets[1], 0: tickets[2]})
            self.assertEqual(batch.active_slots, (0,))
            batch.close_slots([0])
            with self.assertRaisesRegex(ValueError, "unique"):
                batch.reset({1: tickets[0]})
            with self.assertRaises(ValueError):
                batch.reset({1: tickets[1], 2: tickets[1]})

    def test_policy_versions_have_distinct_training_groups(self):
        tickets = self.suite.sampler(shuffle=False).take(2)
        with self.batch() as batch:
            first = batch.reset({0: tickets[0]})[0]
            second = batch.reset({1: tickets[1]}, policy=PolicyVersion("fixture-policy", "checkpoint-12"))[1]
            self.assertEqual(first["rollout"]["task_group_id"], second["rollout"]["task_group_id"])
            self.assertNotEqual(first["rollout"]["group_id"], second["rollout"]["group_id"])

    def test_dataset_revision_changes_group_even_without_suite_metadata(self):
        tickets = self.suite.sampler(shuffle=False).take(2)
        revision = ["revision-1"]
        def factory(spec):
            task = self.make(spec)
            task.data_reference["release_id"] = revision[0]
            return task
        with RolloutBatch(self.root / "run", slots=2, policy=self.policy, task_factory=factory) as batch:
            first = batch.reset({0: tickets[0]})[0]
            revision[0] = "revision-2"
            second = batch.reset({1: tickets[1]})[1]
            self.assertEqual(first["rollout"]["task_group_id"], second["rollout"]["task_group_id"])
            self.assertNotEqual(first["rollout"]["group_id"], second["rollout"]["group_id"])

    def test_episode_time_limits_define_distinct_training_groups(self):
        ticket = next(self.suite.sampler())
        records = []
        for seconds in (1, 120):
            with RolloutBatch(self.root / f"seconds-{seconds}", slots=1, policy=self.policy,
                              episode_seconds=seconds, task_factory=self.make) as batch:
                record = batch.reset({0: ticket})[0]["rollout"]
                self.assertEqual(record["limits"], {"actions": 32, "seconds": seconds})
                persisted = json.loads((batch.root / record["path"] / "rollout.json").read_text())
                self.assertEqual(persisted["limits"], record["limits"])
                records.append(record)
        self.assertEqual(records[0]["task_group_id"], records[1]["task_group_id"])
        self.assertNotEqual(records[0]["group_id"], records[1]["group_id"])

    def test_episode_action_limits_define_distinct_training_groups(self):
        ticket = next(self.suite.sampler())
        records = []
        for limit in (1, 32):
            def factory(spec, max_actions=limit):
                task = self.make(spec)
                task.max_actions = max_actions
                return task
            with RolloutBatch(self.root / f"actions-{limit}", slots=1, policy=self.policy,
                              task_factory=factory) as batch:
                record = batch.reset({0: ticket})[0]["rollout"]
                self.assertEqual(record["limits"], {"actions": limit, "seconds": 120})
                records.append(record)
        self.assertEqual(records[0]["task_group_id"], records[1]["task_group_id"])
        self.assertNotEqual(records[0]["group_id"], records[1]["group_id"])

    def test_result_sidecars_preserve_cleanup_error(self):
        def factory(spec):
            task = self.make(spec)
            task.environment.bad_cleanup = True
            return task
        with RolloutBatch(self.root / "run", slots=1, policy=self.policy, task_factory=factory) as batch:
            batch.reset({0: next(self.suite.sampler())})
            result = batch.step({0: {"submit": {"count": 0}}})[0]
            self.assertEqual(result["status"], "error")
            self.assertIsNone(result["reward"])
        self.assertEqual(batch.summary()["counts"]["error"], 1)

    def test_same_output_run_is_not_reused(self):
        with self.batch():
            pass
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.batch()


if __name__ == "__main__":
    unittest.main()
