import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from agent_gym.contracts import TaskSpec
from agent_gym.episode import Episode
from agent_gym.taskpacks import load_taskpack
from agent_gym.tasks.sql import make_sql_task, matching_rows


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/sql-workload/tasks.jsonl"


def small_payload(**verifier):
    return {"profile": "sql/v1", "instruction": "Return the items.",
            "database": {"ddl": ["CREATE TABLE items (name TEXT, amount REAL)"],
                         "tables": {"items": {"columns": ["name", "amount"], "rows": [["a", 1], ["a", 1], ["b", 2]]}}},
            "verifier": verifier}


class SQLTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def task(self, payload):
        task = make_sql_task(TaskSpec("sql-test"), payload, {"record_id": "sample"}, self.root)
        self.addCleanup(task.environment.close)
        return task

    def test_equivalent_queries_and_multistep_episode(self):
        pack = load_taskpack(EXAMPLE, family="warehouse-sql")
        task = pack.make(pack.specs("train")[0])
        with Episode(task, self.root) as episode:
            initial = episode.initial
            self.assertEqual(initial["data_reference"]["record_id"], "sql:october-region-revenue")
            self.assertNotIn("verifier", json.dumps(initial))
            self.assertNotIn("SUM(i.quantity", json.dumps(initial))
            schema = episode.step({"tool": "database_schema", "arguments": {}})
            self.assertEqual(len(schema["result"]["tables"]), 4)
            query = episode.step({"tool": "query", "arguments": {"sql": "SELECT COUNT(*) FROM orders"}})
            self.assertEqual(query["result"]["rows"], [[8]])
            result = episode.step({"submit": {"sql": "WITH bills AS (SELECT order_id,SUM(quantity*unit_price) revenue FROM order_items GROUP BY order_id) SELECT region,SUM(revenue) total FROM bills JOIN orders USING(order_id) JOIN customers USING(customer_id) WHERE status='completed' AND ordered_at BETWEEN '2026-10-01' AND '2026-10-31' GROUP BY region ORDER BY total DESC,region"}})
            self.assertEqual(result["result"]["status"], "passed")
            events = [json.loads(line) for line in (episode.path / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["content"]["data_reference"]["pack_sha256"], pack.pack_sha256)

    def test_row_order_duplicate_multiplicity_and_partial_credit(self):
        task = self.task(small_payload(sql="SELECT name,amount FROM items", ordered=False))
        self.assertTrue(task.verify({"rows": [["b", 2], ["a", 1], ["a", 1]]}).passed)
        partial = task.verify({"rows": [["a", 1], ["b", 2]]})
        self.assertFalse(partial.passed)
        self.assertAlmostEqual(partial.reward, 0.8)
        ordered = self.task(small_payload(sql="SELECT name,amount FROM items ORDER BY amount", ordered=True))
        self.assertFalse(ordered.verify({"rows": [["b", 2], ["a", 1], ["a", 1]]}).passed)

    def test_tolerant_matching_finds_augmenting_path(self):
        # A greedy match consumes 0.0 for 0.09, leaving the exact 0.0 unmatched.
        matched = matching_rows([[0.09], [0.0]], [[0.0], [0.18]], ordered=False, atol=0.1, rtol=0)
        self.assertEqual(matched, 2)
        task = self.task(small_payload(rows=[[3.0]], atol=1e-6, rtol=0))
        self.assertTrue(task.verify({"rows": [[3.0000001]]}).passed)
        self.assertFalse(task.verify({"rows": [["3.0"]]}).passed)
        self.assertFalse(task.verify({"rows": [[10 ** 500]]}).passed)

    def test_environment_can_move_between_worker_threads(self):
        pack = load_taskpack(EXAMPLE)
        with ThreadPoolExecutor(max_workers=1) as creator:
            task = creator.submit(pack.make, pack.specs("train")[0]).result()
        with Episode(task, self.root) as episode:
            with ThreadPoolExecutor(max_workers=1) as caller:
                response = caller.submit(episode.step, {"tool": "query", "arguments": {"sql": "SELECT COUNT(*) FROM orders"}}).result()
            self.assertEqual(response["result"]["rows"], [[8]])
            self.assertTrue(task.verify({"rows": [["West", 520.5], ["East", 410], ["North", 150]]}).passed)

    def test_failed_query_and_readonly_world(self):
        task = self.task(small_payload(sql="SELECT COUNT(*) FROM items"))
        with self.assertRaisesRegex(ValueError, "SQL query failed"):
            task.environment.call("query", {"sql": "DELETE FROM items"})
        self.assertFalse(task.verify({"sql": "SELECT missing FROM items"}).passed)
        self.assertTrue(task.verify({"rows": [[3]]}).passed)

    def test_query_preview_and_result_limit(self):
        task = self.task(small_payload(sql="SELECT name FROM items"))
        result = task.environment.call("query", {"sql": "SELECT name FROM items", "limit": 1})
        self.assertEqual(result["rows"], [["a"]])
        self.assertTrue(result["truncated"])
        payload = small_payload(rows=[[1]])
        payload["max_result_rows"] = 1
        limited = self.task(payload)
        self.assertFalse(limited.verify({"sql": "SELECT amount FROM items"}).passed)

    def test_sqlite_file_snapshot_and_hash(self):
        file = self.root / "database.sqlite"
        with sqlite3.connect(file) as db:
            db.execute("CREATE TABLE observed (x INTEGER)")
            db.executemany("INSERT INTO observed VALUES (?)", [(3,), (4,)])
        payload = small_payload(sql="SELECT SUM(x) FROM observed")
        payload["database"] = {"path": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        task = self.task(payload)
        with sqlite3.connect(file) as db:
            db.execute("INSERT INTO observed VALUES (999)")
        self.assertTrue(task.verify({"sql": "SELECT SUM(x) FROM observed"}).passed)
        self.assertTrue(task.verify({"rows": [[7]]}).passed)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.task(payload)

    def test_column_contract_and_empty_result(self):
        task = self.task(small_payload(sql="SELECT name FROM items WHERE 0", columns=["name"]))
        self.assertTrue(task.verify({"sql": "SELECT name FROM items WHERE amount>100"}).passed)
        self.assertFalse(task.verify({"sql": "SELECT amount FROM items WHERE 0"}).passed)
        self.assertTrue(task.verify({"rows": [], "columns": ["name"]}).passed)

    def test_sql_budget(self):
        payload = small_payload(rows=[[1]])
        payload["query_vm_steps"] = 1000
        task = self.task(payload)
        with self.assertRaisesRegex(ValueError, "interrupted"):
            task.environment.call("query", {"sql": "WITH RECURSIVE t(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM t) SELECT SUM(x) FROM t"})

    def test_pack_population_order_and_splits(self):
        pack = load_taskpack(EXAMPLE, family="warehouse-sql")
        reversed_file = self.root / "reversed.jsonl"
        reversed_file.write_text("\n".join(reversed(EXAMPLE.read_text().splitlines())) + "\n")
        reversed_pack = load_taskpack(reversed_file, family="warehouse-sql")
        self.assertEqual(pack.pack_sha256, reversed_pack.pack_sha256)
        self.assertEqual(pack.specs(), reversed_pack.specs())
        self.assertEqual(pack.describe()["splits"], {"train": 2, "validation": 1, "test": 1})
        with self.assertRaisesRegex(ValueError, "finite task population"):
            pack.make(TaskSpec("warehouse-sql", seed=2))

    def test_plain_task_form(self):
        value = {"schema_version": 1, "id": "small", "source": {"uri": "example://small", "revision": "1"},
                 "split": "test", "task": small_payload(sql="SELECT SUM(amount) FROM items")}
        path = self.root / "tasks.jsonl"
        path.write_text(json.dumps(value) + "\n")
        pack = load_taskpack(path)
        task = pack.make(pack.specs()[0])
        self.addCleanup(task.environment.close)
        self.assertTrue(task.verify({"rows": [[4]]}).passed)

    def test_actual_data_engineering_release(self):
        try:
            from ai_data import build
        except ImportError:
            self.skipTest("Data Engineering package is not installed")
        release = self.root / "release"
        manifest = build(EXAMPLE.parent / "data.recipe.json", release)
        pack = load_taskpack(release)
        self.assertEqual(pack.release_id, manifest["release_id"])
        self.assertEqual(pack.describe()["splits"], {"train": 2, "validation": 1, "test": 1})
        task = pack.make(pack.specs("validation")[0])
        self.addCleanup(task.environment.close)
        self.assertTrue(task.verify({"rows": [["Optics", 631.5], ["Electronics", 600], ["Mechanics", 240]]}).passed)
        target = release / "test.jsonl"
        target.write_text(target.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "partition hash mismatch"):
            load_taskpack(release)


if __name__ == "__main__":
    unittest.main()
