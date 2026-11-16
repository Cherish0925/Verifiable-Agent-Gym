"""Export factual trajectories, not invented chain-of-thought or token probabilities."""

import json
from pathlib import Path

from .storage import dumps
from .contracts import TaskSpec


def export_run(root: Path, destination: Path, *, include_unscored=False) -> int:
    root, destination = Path(root).resolve(), Path(destination).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("x", encoding="utf-8", newline="\n") as output:
        for result in summary["results"]:
            if not include_unscored and (result["status"] not in ("passed", "failed") or result["reward"] is None):
                continue
            if not result.get("attempt_path"):
                continue
            # Task ID and attempt ID survive moves between Windows and Unix.
            attempt = root / TaskSpec(**result["task"]).id / result["episode_id"]
            events = [json.loads(line) for line in (attempt / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            record = {"schema_version": 1, "episode_id": result["episode_id"], "task": result["task"], "agent": result["agent"],
                      "status": result["status"], "reward": result["reward"], "verification": result["verification"],
                      "trajectory": [e for e in events if e["kind"] in ("observation", "action")]}
            output.write(dumps(record) + "\n")
            count += 1
    return count
