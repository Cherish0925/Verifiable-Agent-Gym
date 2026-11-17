import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/sql-workload/tasks.jsonl"
FAMILY = "cli-warehouse-sql"


def cli_environment():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_cli(*args, cwd):
    return subprocess.run([sys.executable, "-m", "agent_gym", *map(str, args)], cwd=cwd,
                          env=cli_environment(), capture_output=True, text=True, encoding="utf-8", timeout=30)


class TaskPackCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gym task cli ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = self.root / "task data = α.jsonl"
        shutil.copyfile(FIXTURE, self.pack)

    def test_pack_info_and_isolated_registration(self):
        info = run_cli("pack-info", self.pack.name, "--family", FAMILY, cwd=self.root)
        self.assertEqual(info.returncode, 0, info.stderr)
        value = json.loads(info.stdout)
        self.assertEqual(value["family"], FAMILY)
        self.assertEqual(value["profile"], "sql/v1")
        self.assertEqual(value["splits"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(len(value["pack_sha256"]), 64)
        before = run_cli("list", cwd=self.root)
        loaded = run_cli("--task-pack", f"{FAMILY}={self.pack.name}", "list", cwd=self.root)
        after = run_cli("list", cwd=self.root)
        self.assertEqual(loaded.returncode, 0, loaded.stderr)
        self.assertNotIn(FAMILY, before.stdout.splitlines())
        self.assertIn(FAMILY, loaded.stdout.splitlines())
        self.assertNotIn(FAMILY, after.stdout.splitlines())

    def test_generate_preserves_task_identity_and_split(self):
        target = self.root / "generated.jsonl"
        result = run_cli("--task-pack", f"{FAMILY}={self.pack.name}", "generate", "--tasks", FAMILY,
                         "--count", "2", "--difficulty", "2", "--split", "train", "--output", target,
                         cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual({row["spec"]["seed"] for row in records}, {0, 1})
        self.assertEqual({row["observation"]["data_reference"]["record_id"] for row in records},
                         {"sql:october-region-revenue", "sql:repeat-customer-spend"})
        for record in records:
            self.assertEqual(record["spec"]["family"], FAMILY)
            self.assertEqual(record["spec"]["split"], "train")
            self.assertEqual(record["observation"]["data_reference"]["split"], "train")
            self.assertNotIn("verifier", record["observation"])

    def test_default_generate_uses_finite_population_and_native_difficulty(self):
        target = self.root / "native.jsonl"
        result = run_cli("--task-pack", f"{FAMILY}={self.pack.name}", "generate", "--tasks", FAMILY,
                         "--output", target, cwd=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual({row["spec"]["seed"] for row in records}, {0, 1})
        self.assertEqual({row["spec"]["difficulty"] for row in records}, {2})
        self.assertEqual({row["spec"]["split"] for row in records}, {"train"})

    def test_excess_count_fails_before_creating_output(self):
        target = self.root / "too-many.jsonl"
        result = run_cli("--task-pack", f"{FAMILY}={self.pack.name}", "generate", "--tasks", FAMILY,
                         "--count", "3", "--output", target, cwd=self.root)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertFalse(target.exists(), "an invalid population request must not leave partial generated tasks")

    def test_invalid_pack_argument_has_actionable_error(self):
        result = run_cli("--task-pack", "missing-family-assignment", "list", cwd=self.root)
        self.assertEqual(result.returncode, 2)
        self.assertIn("FAMILY=PATH", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "MCP extra not installed")
    def test_connect_keeps_absolute_pack_path_and_argument_order(self):
        for host in ("mcp", "research-os"):
            config = self.root / f"{host}.json"
            result = run_cli("--task-pack", f"{FAMILY}={self.pack.name}", "connect", "--host", host,
                             "--output", config, cwd=self.root)
            self.assertEqual(result.returncode, 0, result.stderr)
            server = json.loads(config.read_text(encoding="utf-8"))["mcpServers"]["agent-gym"]
            self.assertEqual(server["command"], sys.executable)
            self.assertEqual(server["args"], ["-m", "agent_gym", "--task-pack", f"{FAMILY}={self.pack.resolve()}", "mcp"])
            if host == "research-os":
                self.assertEqual(server["transport"], "stdio")


@unittest.skipUnless(importlib.util.find_spec("mcp"), "MCP extra not installed")
class TaskPackMCPConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_configuration_loads_task_from_another_directory(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory(prefix="gym task mcp ") as directory:
            root = Path(directory)
            pack = root / "task data = α.jsonl"
            shutil.copyfile(FIXTURE, pack)
            config = root / "server.json"
            connected = run_cli("--task-pack", f"{FAMILY}={pack.name}", "connect", "--output", config, cwd=root)
            self.assertEqual(connected.returncode, 0, connected.stderr)
            server = json.loads(config.read_text(encoding="utf-8"))["mcpServers"]["agent-gym"]
            launch = root / "different-working-directory"
            launch.mkdir()
            env = cli_environment()
            env["AGENT_GYM_RUNS"] = str(root / "episodes")
            params = StdioServerParameters(command=server["command"], args=server["args"], env=env, cwd=str(launch))
            async with asyncio.timeout(30):
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as client:
                        await client.initialize()

                        async def call(name, arguments):
                            result = await client.call_tool(name, arguments)
                            self.assertFalse(result.isError, result)
                            return result.structuredContent or json.loads(result.content[0].text)

                        families = await call("gym_tasks", {})
                        self.assertIn(FAMILY, families["families"])
                        initial = await call("gym_open", {"family": FAMILY, "seed": 0, "difficulty": 2, "split": "train"})
                        self.assertEqual(initial["data_reference"]["record_id"], "sql:october-region-revenue")
                        response = await call("gym_action", {"episode_id": initial["episode_id"],
                            "action": {"tool": "query", "arguments": {"sql": "SELECT COUNT(*) FROM orders"}}})
                        self.assertEqual(response["result"]["rows"], [[8]])
                        closed = await call("gym_close", {"episode_id": initial["episode_id"]})
                        self.assertEqual(closed["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
