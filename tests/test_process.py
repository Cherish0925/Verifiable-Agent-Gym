import json
from pathlib import Path
import sys
import tempfile
import unittest

from agent_gym.contracts import TaskSpec
from agent_gym.process import CommandAgent
from agent_gym.runner import run_batch


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gym process ")
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_parallel_jsonl_processes(self):
        command = [sys.executable, "-m", "agent_gym.reference"]
        specs = [TaskSpec(f) for f in ("evidence-aggregation", "rc-identification", "thin-film-design")]
        report = run_batch(specs, self.root, lambda path: CommandAgent(command, path), agent_name="command", jobs=3)
        self.assertEqual(report["counts"]["passed"], 3, report)

    def test_timeout_reaps_agent(self):
        agent = CommandAgent([sys.executable, "-c", "import time; time.sleep(30)"], self.root)
        self.addCleanup(agent.close)
        with self.assertRaises(TimeoutError):
            agent.act({"type": "start"}, timeout=0.1)
        self.assertIsNotNone(agent.process.poll())

    def test_malformed_output_is_not_a_verdict(self):
        agent = CommandAgent([sys.executable, "-c", "import sys; sys.stdin.readline(); print('not json', flush=True)"], self.root)
        self.addCleanup(agent.close)
        with self.assertRaises(json.JSONDecodeError):
            agent.act({}, timeout=5)

    def test_unicode_and_space_in_paths(self):
        agent = CommandAgent([sys.executable, "-c", "import json,sys; x=json.loads(sys.stdin.readline()); print(json.dumps({'submit':x}),flush=True)"], self.root / "研究 workspace")
        self.addCleanup(agent.close)
        self.assertEqual(agent.act({"text": "光学 µm"}, timeout=5), {"submit": {"text": "光学 µm"}})


if __name__ == "__main__":
    unittest.main()
