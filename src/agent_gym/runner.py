"""Bounded episode scheduling with recoverable, individually owned attempts."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import math
from pathlib import Path
import threading
from typing import Callable

from .contracts import Agent, Json, TaskSpec
from . import __version__
from .episode import Episode
from .registry import make
from .storage import RunLease, atomic_json


def run_episode(spec: TaskSpec, root: Path, agent_factory: Callable[[Path], Agent], *, agent_name: str, seconds: float = 120, action_seconds: float = 30, cancel: threading.Event | None = None) -> Json:
    episode = Episode(make(spec), root, seconds=seconds, agent=agent_name)
    try:
        agent = agent_factory(episode.path / "workspace")
        episode.resources.callback(agent.close)
        observation = episode.initial
        while episode.result is None:
            if cancel is not None and cancel.is_set():
                episode.stop("cancelled", "Batch interrupted")
                break
            if episode.remaining_seconds <= 0:
                episode.stop("truncated", "Episode time limit")
                break
            action = agent.act(observation, timeout=min(action_seconds, episode.remaining_seconds))
            observation = episode.step(action)
    except TimeoutError as exc:
        episode.stop("truncated", str(exc))
    except Exception as exc:
        episode.stop("error", f"Agent error: {type(exc).__name__}: {exc}")
    finally:
        if episode.result is None:
            episode.stop()
    return {**episode.result, "attempt_path": str(episode.path)}


def _latest(root: Path, spec: TaskSpec) -> Json | None:
    directory = root / spec.id
    candidates = sorted(directory.glob("*/result.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["task"] == spec.to_dict():
            return {**result, "attempt_path": str(path.parent)}
    return None


def summarize(results: list[Json]) -> Json:
    statuses = {status: sum(r["status"] == status for r in results) for status in ("passed", "failed", "truncated", "error", "cancelled")}
    scored = [r for r in results if r["status"] in ("passed", "failed") and r["reward"] is not None]
    return {"total": len(results), "counts": statuses, "scored": len(scored),
            "success_rate_scored": statuses["passed"] / len(scored) if scored else None,
            "success_rate_all": statuses["passed"] / len(results) if results else None,
            "mean_reward_scored": sum(r["reward"] for r in scored) / len(scored) if scored else None}


def run_batch(specs: list[TaskSpec], root: Path, agent_factory: Callable[[Path], Agent], *, agent_name="reference", agent_config=None, jobs=4, seconds=120, action_seconds=30, resume=False, retry_errors=False, progress=None) -> Json:
    if jobs < 1 or any(not math.isfinite(t) or t <= 0 for t in (seconds, action_seconds)):
        raise ValueError("jobs and time limits must be finite and positive")
    if len({spec.id for spec in specs}) != len(specs):
        raise ValueError("Task specifications must be unique within a batch")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = {"schema_version": 1, "gym_version": __version__, "tasks": [spec.to_dict() for spec in specs], "agent": agent_name,
              "agent_config": agent_config or {}, "seconds": seconds, "action_seconds": action_seconds}
    results = {}
    cancel = threading.Event()
    with RunLease(root):
        manifest = root / "run.json"
        if manifest.exists():
            if not resume:
                raise ValueError("Output already contains a run; use --resume or a new output directory")
            if json.loads(manifest.read_text(encoding="utf-8")) != config:
                raise ValueError("Resume requires the same tasks, agent configuration, and limits")
        else:
            atomic_json(manifest, config)
        todo = []
        for spec in specs:
            previous = _latest(root, spec) if resume else None
            if previous and previous["status"] in ("passed", "failed", "truncated"):
                results[spec.id] = previous
            elif previous and previous["status"] == "error" and not retry_errors:
                results[spec.id] = previous
            else:
                todo.append(spec)
        pending_specs = iter(todo)
        futures = {}
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="gym") as pool:
            def schedule():
                while len(futures) < jobs and not cancel.is_set():
                    spec = next(pending_specs, None)
                    if spec is None:
                        break
                    future = pool.submit(run_episode, spec, root, agent_factory, agent_name=agent_name, seconds=seconds, action_seconds=action_seconds, cancel=cancel)
                    futures[future] = spec
            schedule()
            try:
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        spec = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            # Factory/setup failure has no valid reward or environment trace.
                            result = {"task": spec.to_dict(), "status": "error", "reward": None, "reason": f"Setup/cleanup error: {type(exc).__name__}: {exc}"}
                            failures = root / "setup-errors"
                            failures.mkdir(exist_ok=True)
                            atomic_json(failures / f"{spec.id}.json", result)
                        results[spec.id] = result
                        if progress:
                            progress(spec, result)
                    schedule()
            except KeyboardInterrupt:
                cancel.set()
                for future, spec in list(futures.items()):
                    try:
                        results[spec.id] = future.result()
                    except Exception as exc:
                        results[spec.id] = {"task": spec.to_dict(), "status": "error", "reward": None, "reason": str(exc)}
        ordered = [results[spec.id] for spec in specs if spec.id in results]
        report = {**summarize(ordered), "requested": len(specs), "not_started": len(specs) - len(ordered), "interrupted": cancel.is_set(), "results": ordered}
        atomic_json(root / "summary.json", report)
        return report
