from __future__ import annotations

from datetime import datetime, timezone
from contextlib import ExitStack
import math
from pathlib import Path
import platform
import threading
import time
import uuid
from typing import Any

from .contracts import Json, Task
from . import __version__
from .storage import atomic_json, dumps, json_copy


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Episode:
    """A single owned environment, ordered actions, and one final verification."""

    def __init__(self, task: Task, root: Path, *, seconds: float = 120, agent: str = "interactive"):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("episode seconds must be finite and positive")
        if task.max_actions < 1:
            raise ValueError("a task must allow at least one action")
        self.task = task
        self.id = uuid.uuid4().hex
        self.path = Path(root) / task.spec.id / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.seconds = seconds
        self.steps = 0
        self.result: Json | None = None
        self.resources = ExitStack()
        self.resources.callback(task.environment.close)
        self._lock = threading.RLock()
        self._events = (self.path / "events.jsonl").open("a", encoding="utf-8", newline="\n")
        self.metadata = {
            "schema_version": 1, "gym_version": __version__, "episode_id": self.id,
            "task": task.spec.to_dict(), "agent": agent, "started_at": now(),
            "python": platform.python_version(), "platform": platform.system(),
            "limits": {"actions": task.max_actions, "seconds": seconds},
        }
        atomic_json(self.path / "episode.json", self.metadata)
        self.initial = json_copy(task.observation())
        self.initial["episode_id"] = self.id
        self.initial["remaining_actions"] = task.max_actions
        self._record("observation", self.initial)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.seconds - (time.monotonic() - self.started))

    def _record(self, kind: str, content: Any) -> None:
        self._events.write(dumps({"time": now(), "step": self.steps, "kind": kind, "content": content}) + "\n")
        self._events.flush()

    def _finish(self, status: str, *, verification: Json | None = None, reason: str = "") -> Json:
        if self.result is not None:
            return json_copy(self.result)
        self.result = {
            **self.metadata, "status": status, "steps": self.steps, "finished_at": now(),
            "elapsed_seconds": time.monotonic() - self.started,
            "reward": verification["reward"] if verification is not None else None,
            "verification": verification, "reason": reason,
        }
        try:
            self.resources.close()
        except Exception as exc:
            self.result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            self.result["status"] = "error"
            self.result["reward"] = None
        self._record("result", self.result)
        atomic_json(self.path / "result.json", self.result)
        self._events.close()
        return json_copy(self.result)

    def stop(self, status: str = "cancelled", reason: str = "Closed by caller") -> Json:
        if status not in ("cancelled", "error", "truncated"):
            raise ValueError("stop status must be cancelled, error, or truncated")
        with self._lock:
            return self._finish(status, reason=reason)

    def step(self, action: Json) -> Json:
        with self._lock:
            if self.result is not None:
                raise ValueError("Episode is finished; open a new episode")
            if self.remaining_seconds <= 0:
                return {"type": "terminal", "result": self._finish("truncated", reason="Episode time limit")}
            self.steps += 1
            try:
                action = json_copy(action)
            except (TypeError, ValueError) as exc:
                action = {"invalid_action": str(exc)}
            self._record("action", action)
            try:
                if not isinstance(action, dict):
                    raise ValueError("Action must be a JSON object")
                if set(action) == {"submit"}:
                    try:
                        verdict = self.task.verify(action["submit"])
                        verification = json_copy(verdict.to_dict())
                    except Exception as exc:
                        result = self._finish("error", reason=f"Verifier error: {type(exc).__name__}: {exc}")
                    else:
                        result = self._finish("passed" if verdict.passed else "failed", verification=verification)
                    return {"type": "terminal", "result": result}
                if set(action) != {"tool", "arguments"} or not isinstance(action["arguments"], dict):
                    raise ValueError('Use {"tool": "name", "arguments": {...}} or {"submit": {...}}')
                if not isinstance(action["tool"], str) or action["tool"] not in {tool.name for tool in self.task.environment.tools}:
                    raise ValueError(f"Unknown tool: {action['tool']}")
                value = self.task.environment.call(action["tool"], action["arguments"])
                try:
                    value = json_copy(value)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("Environment returned a non-JSON result") from exc
                observation = {"type": "tool_result", "tool": action["tool"], "result": value}
            except ValueError as exc:
                observation = {"type": "tool_error", "error": str(exc)}
            except Exception as exc:
                return {"type": "terminal", "result": self._finish("error", reason=f"Environment error: {type(exc).__name__}: {exc}")}
            observation["remaining_actions"] = self.task.max_actions - self.steps
            self._record("observation", observation)
            if self.steps >= self.task.max_actions or self.remaining_seconds <= 0:
                reason = "Action limit" if self.steps >= self.task.max_actions else "Episode time limit"
                return {"type": "terminal", "result": self._finish("truncated", reason=reason), "observation": observation}
            return observation

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _):
        if self.result is None:
            self.stop("error" if exc else "cancelled", str(exc) if exc else "Episode closed without submission")
