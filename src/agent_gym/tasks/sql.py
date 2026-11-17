"""Data-backed relational tasks with result-based verification."""

from __future__ import annotations

from collections import Counter, deque
from contextlib import closing
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

from ..contracts import Task, TaskSpec, Tool, Verdict, arguments_only, object_schema
from ..storage import json_copy


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("database identifiers must be nonempty strings")
    return '"' + value.replace('"', '""') + '"'


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(value: Any) -> list[list]:
    if not isinstance(value, list) or any(not isinstance(row, (list, tuple)) for row in value):
        raise ValueError("rows must be a list of rows")
    result = []
    width = None
    for row in value:
        if width is not None and len(row) != width:
            raise ValueError("rows must have a consistent number of columns")
        width = len(row)
        cells = []
        for cell in row:
            if cell is None or isinstance(cell, str) or type(cell) is int:
                cells.append(cell)
            elif type(cell) is float and math.isfinite(cell):
                cells.append(cell)
            elif type(cell) is bool:
                cells.append(int(cell))
            else:
                raise ValueError("SQL result cells must be finite numbers, strings or null")
        result.append(cells)
    return result


def _readonly(action, arg1, arg2, _database, _trigger):
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _execute(connection: sqlite3.Connection, sql: str, *, limit: int, seconds: float, vm_steps: int):
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("sql must be a nonempty SQL statement")
    deadline, callbacks = time.monotonic() + seconds, 0

    def progress():
        nonlocal callbacks
        callbacks += 1
        return int(callbacks * 1000 > vm_steps or time.monotonic() > deadline)

    connection.set_progress_handler(progress, 1000)
    try:
        cursor = connection.execute(sql)
        if cursor.description is None:
            raise ValueError("query must return a table")
        columns = [column[0] for column in cursor.description]
        result = cursor.fetchmany(limit + 1)
        return columns, _rows(result[:limit]), len(result) > limit
    except sqlite3.Error as exc:
        raise ValueError(f"SQL query failed: {exc}") from exc
    finally:
        connection.set_progress_handler(None, 0)


def _same_cell(a, b, *, atol: float, rtol: float):
    if type(a) in (int, float) and type(b) in (int, float):
        if a == b:
            return True
        try:
            return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
        except OverflowError:
            return False
    return type(a) is type(b) and a == b


def matching_rows(actual: list[list], expected: list[list], *, ordered: bool, atol: float, rtol: float) -> int:
    def same(a, b):
        return len(a) == len(b) and all(_same_cell(x, y, atol=atol, rtol=rtol) for x, y in zip(a, b))

    if ordered:
        return sum(same(a, b) for a, b in zip(actual, expected))
    if atol == rtol == 0:
        # Counter preserves duplicate multiplicity; int/float equality matches SQLite numerical results.
        return sum((Counter(map(tuple, actual)) & Counter(map(tuple, expected))).values())
    adjacency = [[j for j, b in enumerate(expected) if same(a, b)] for a in actual]
    left, right = [-1] * len(actual), [-1] * len(expected)
    matches = 0
    # Augmenting paths avoid greedy tolerance matching and recursive depth limits.
    for origin in range(len(actual)):
        queue, visited, predecessor = deque([origin]), {origin}, {}
        endpoint = None
        while queue and endpoint is None:
            i = queue.popleft()
            for j in adjacency[i]:
                if j in predecessor:
                    continue
                predecessor[j] = i
                if right[j] == -1:
                    endpoint = j
                    break
                if right[j] not in visited:
                    visited.add(right[j])
                    queue.append(right[j])
        if endpoint is not None:
            matches += 1
            j = endpoint
            while j != -1:
                i = predecessor[j]
                previous = left[i]
                left[i], right[j] = j, i
                j = previous
    return matches


class SQLDatabase:
    """One immutable database snapshot per episode, with bounded query results."""

    tools = (
        Tool("database_schema", "Inspect relational tables, columns and schema.", object_schema({})),
        Tool("query", "Run a read-only SQL statement and inspect its result table.", object_schema({
            "sql": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
        }, ["sql"])),
    )

    def __init__(self, database: dict, asset_root: Path, *, seconds: float = 2.0, vm_steps: int = 1_000_000):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.lock = threading.RLock()
        self.seconds, self.vm_steps = seconds, vm_steps
        self.closed = False
        try:
            if set(database) == {"path", "sha256"}:
                relative = database["path"]
                if not isinstance(relative, str) or Path(relative).is_absolute():
                    raise ValueError("database path must be relative to asset_root")
                path = (asset_root / relative).resolve()
                if not path.is_relative_to(asset_root.resolve()):
                    raise ValueError("database path escapes asset_root")
                if _hash_file(path) != database["sha256"]:
                    raise ValueError("database file hash mismatch")
                with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
                    source.backup(self.connection)
                if _hash_file(path) != database["sha256"]:
                    raise ValueError("database changed while loading")
            elif set(database) == {"ddl", "tables"}:
                if not isinstance(database["ddl"], list) or not isinstance(database["tables"], dict):
                    raise ValueError("database ddl must be a list and tables an object")
                for statement in database["ddl"]:
                    if not isinstance(statement, str) or not re.match(r"^\s*CREATE\s+(?:UNIQUE\s+)?(?:TABLE|VIEW|INDEX)\b", statement, re.I):
                        raise ValueError("ddl entries must be single CREATE TABLE, VIEW or INDEX statements")
                    self.connection.execute(statement)
                for name, table in database["tables"].items():
                    if not isinstance(table, dict) or set(table) != {"columns", "rows"}:
                        raise ValueError("each table needs columns and rows")
                    columns = table["columns"]
                    if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns):
                        raise ValueError("columns must be a nonempty list of distinct names")
                    rows = _rows(table["rows"])
                    if any(len(row) != len(columns) for row in rows):
                        raise ValueError("inserted row width differs from columns")
                    sql = f"INSERT INTO {_identifier(name)} ({','.join(_identifier(c) for c in columns)}) VALUES ({','.join('?' for _ in columns)})"
                    self.connection.executemany(sql, rows)
            else:
                raise ValueError("database requires path/sha256 or ddl/tables")
            self.connection.commit()
            tables = self.connection.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
            self.schema = {"tables": [{"name": name, "type": kind, "ddl": sql,
                            "columns": [{"name": c[1], "type": c[2], "not_null": bool(c[3]), "primary_key": bool(c[5])}
                                        for c in self.connection.execute(f"PRAGMA table_info({_identifier(name)})").fetchall()]}
                           for name, kind, sql in tables]}
            self.connection.execute("PRAGMA query_only=ON")
            self.connection.set_authorizer(_readonly)
        except Exception:
            self.connection.close()
            self.closed = True
            raise

    def execute(self, sql: str, *, limit: int):
        with self.lock:
            if self.closed:
                raise ValueError("database episode is closed")
            return _execute(self.connection, sql, limit=limit, seconds=self.seconds, vm_steps=self.vm_steps)

    def call(self, name: str, arguments: dict):
        if name == "database_schema":
            arguments_only(arguments)
            return json_copy(self.schema)
        if name == "query":
            arguments_only(arguments, "sql", "limit")
            limit = arguments.get("limit", 200)
            if type(limit) is not int or not 1 <= limit <= 1000:
                raise ValueError("query limit must be an integer in [1, 1000]")
            columns, rows, truncated = self.execute(arguments.get("sql"), limit=limit)
            return {"columns": columns, "rows": rows, "truncated": truncated}
        raise ValueError(f"unknown tool: {name}")

    def close(self):
        with self.lock:
            if not self.closed:
                self.connection.close()
                self.closed = True


@dataclass
class DatasetTask(Task):
    data_reference: dict = field(default_factory=dict)

    def observation(self):
        observation = super().observation()
        observation["data_reference"] = json_copy(self.data_reference)
        return observation


def make_sql_task(spec: TaskSpec, payload: dict, reference: dict, asset_root: Path) -> DatasetTask:
    if not isinstance(payload, dict) or payload.get("profile") != "sql/v1":
        raise ValueError("task payload must use profile sql/v1")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task instruction must be nonempty")
    verifier = payload.get("verifier")
    if not isinstance(verifier, dict) or ("sql" in verifier) == ("rows" in verifier):
        raise ValueError("verifier requires exactly one of sql or rows")
    if set(verifier) - {"sql", "rows", "ordered", "atol", "rtol", "columns"}:
        raise ValueError("unknown verifier option")
    ordered, atol, rtol = verifier.get("ordered", False), verifier.get("atol", 0.0), verifier.get("rtol", 0.0)
    if type(ordered) is not bool:
        raise ValueError("ordered must be boolean")
    for value in (atol, rtol):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("numeric tolerances must be finite and non-negative")
    max_rows = payload.get("max_result_rows", 1000)
    max_actions = payload.get("max_actions", 32)
    seconds = payload.get("query_seconds", 2.0)
    vm_steps = payload.get("query_vm_steps", 1_000_000)
    if type(max_rows) is not int or not 1 <= max_rows <= 10_000 or type(max_actions) is not int or max_actions < 1:
        raise ValueError("invalid row or action limit")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0 or type(vm_steps) is not int or vm_steps < 1000:
        raise ValueError("query limits must be finite and positive")
    database = payload.get("database")
    if not isinstance(database, dict):
        raise ValueError("database must be an object")
    # Reference answers are computed before agent interaction on a separate connection.
    oracle = SQLDatabase(database, asset_root, seconds=seconds, vm_steps=vm_steps)
    try:
        if "sql" in verifier:
            _, expected, truncated = oracle.execute(verifier["sql"], limit=max_rows)
            if truncated:
                raise ValueError("reference result exceeds max_result_rows")
        else:
            expected = _rows(verifier["rows"])
            if len(expected) > max_rows:
                raise ValueError("reference result exceeds max_result_rows")
    finally:
        oracle.close()
    required_columns = verifier.get("columns")
    if required_columns is not None and (not isinstance(required_columns, list) or not all(isinstance(c, str) for c in required_columns)):
        raise ValueError("verifier columns must be a list of strings")
    environment = SQLDatabase(database, asset_root, seconds=seconds, vm_steps=vm_steps)

    def verify(answer: Any) -> Verdict:
        try:
            if not isinstance(answer, dict):
                raise ValueError("submit sql or rows")
            if set(answer) == {"sql"}:
                columns, actual, truncated = environment.execute(answer["sql"], limit=max_rows)
                if truncated:
                    raise ValueError("submitted result exceeds max_result_rows")
            elif set(answer).issubset({"rows", "columns"}) and "rows" in answer:
                actual, columns = _rows(answer["rows"]), answer.get("columns")
                if len(actual) > max_rows:
                    raise ValueError("submitted result exceeds max_result_rows")
            else:
                raise ValueError("submit exactly sql, or rows with optional columns")
            if required_columns is not None and columns != required_columns:
                return Verdict(False, 0.0, "column names differ from the requested result")
            matched = matching_rows(actual, expected, ordered=ordered, atol=atol, rtol=rtol)
            total = len(actual) + len(expected)
            reward = 2 * matched / total if total else 1.0
            passed = matched == len(actual) == len(expected)
            return Verdict(passed, reward, "result matches" if passed else "result differs",
                           {"matched_rows": matched, "expected_rows": len(expected), "actual_rows": len(actual), "ordered": ordered})
        except ValueError as exc:
            return Verdict(False, 0.0, str(exc))

    semantics = "in the requested order" if ordered else "in any row order, preserving duplicate rows"
    prompt = instruction + f"\nInspect database_schema and use query as needed. Submit {{\"sql\": \"...\"}} or {{\"rows\": [[...]]}}. Results are compared {semantics}."
    if required_columns is not None:
        prompt += f" Required column names: {required_columns}."
    return DatasetTask(spec, prompt, environment, verify, max_actions, json_copy(reference))
