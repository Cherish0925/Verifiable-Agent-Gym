"""Versioned finite task distributions and restartable rollout-ticket sampling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .contracts import Json, TaskSpec
from .storage import json_copy


def fingerprint(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class RolloutTicket:
    """One requested policy attempt; environment conditions come from spec."""

    suite_id: str
    spec: TaskSpec
    group_id: str
    rollout_id: str
    attempt: int
    epoch: int
    sample_seed: int

    def to_dict(self) -> Json:
        return {"suite_id": self.suite_id, "task": self.spec.to_dict(), "group_id": self.group_id,
                "rollout_id": self.rollout_id, "attempt": self.attempt, "epoch": self.epoch,
                "sample_seed": self.sample_seed}


class TaskSuite:
    """A finite set of distinct tasks, with several policy attempts per task."""

    def __init__(self, specs: Iterable[TaskSpec], *, attempts_per_task: int = 1,
                 name: str = "suite", metadata: Json | None = None):
        _integer(attempts_per_task, "attempts_per_task", 1)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("suite name must be nonempty")
        self._specs = tuple(specs)
        if not self._specs or any(not isinstance(spec, TaskSpec) for spec in self._specs):
            raise ValueError("suite must contain TaskSpec instances")
        if len({spec.id for spec in self._specs}) != len(self._specs):
            raise ValueError("suite tasks must be unique; use attempts_per_task for repeat rollouts")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("suite metadata must be an object")
        self._definition = json_copy({"schema_version": 1, "name": name,
                                      "tasks": [spec.to_dict() for spec in self._specs],
                                      "attempts_per_task": attempts_per_task, "metadata": metadata or {}})
        self._suite_id = fingerprint(self._definition)

    @property
    def suite_id(self) -> str:
        return self._suite_id

    @property
    def specs(self) -> tuple[TaskSpec, ...]:
        return self._specs

    @property
    def attempts_per_task(self) -> int:
        return self._definition["attempts_per_task"]

    def to_dict(self) -> Json:
        return {**json_copy(self._definition), "suite_id": self.suite_id}

    def save(self, path: str | Path) -> None:
        with Path(path).open("x", encoding="utf-8", newline="\n") as output:
            json.dump(self.to_dict(), output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")

    @classmethod
    def from_dict(cls, value: Json) -> "TaskSuite":
        if not isinstance(value, dict) or set(value) != {"schema_version", "name", "tasks", "attempts_per_task", "metadata", "suite_id"}:
            raise ValueError("invalid task suite fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported task suite schema")
        suite = cls((TaskSpec(**spec) for spec in value["tasks"]), attempts_per_task=value["attempts_per_task"],
                    name=value["name"], metadata=value["metadata"])
        if suite.suite_id != value["suite_id"]:
            raise ValueError("suite definition does not match its suite_id")
        return suite

    @classmethod
    def load(cls, path: str | Path) -> "TaskSuite":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def sampler(self, *, seed: int = 0, epoch: int = 0, rank: int = 0,
                world_size: int = 1, shuffle: bool = True) -> "SuiteSampler":
        return SuiteSampler(self, seed=seed, epoch=epoch, rank=rank, world_size=world_size, shuffle=shuffle)


class SuiteSampler:
    """Shard whole task groups and checkpoint task issuance, not live worlds."""

    def __init__(self, suite: TaskSuite, *, seed: int = 0, epoch: int = 0,
                 rank: int = 0, world_size: int = 1, shuffle: bool = True):
        for value, name, minimum in ((seed, "seed", 0), (epoch, "epoch", 0), (rank, "rank", 0), (world_size, "world_size", 1)):
            _integer(value, name, minimum)
        if rank >= world_size or type(shuffle) is not bool:
            raise ValueError("rank must be below world_size and shuffle must be boolean")
        self.suite = suite
        self._config = {"schema_version": 1, "suite_id": suite.suite_id, "seed": seed, "epoch": epoch,
                        "rank": rank, "world_size": world_size, "shuffle": shuffle}
        order = list(range(len(suite.specs)))
        if shuffle:
            order.sort(key=lambda index: fingerprint({"seed": seed, "epoch": epoch,
                                                       "suite_id": suite.suite_id, "task_id": suite.specs[index].id}))
        self._order = tuple(order[rank::world_size])
        self._cursor = 0

    @property
    def total(self) -> int:
        return len(self._order) * self.suite.attempts_per_task

    def __len__(self) -> int:
        return self.total - self._cursor

    def __iter__(self):
        return self

    def __next__(self) -> RolloutTicket:
        if self._cursor >= self.total:
            raise StopIteration
        group_index, attempt = divmod(self._cursor, self.suite.attempts_per_task)
        spec = self.suite.specs[self._order[group_index]]
        group = fingerprint({"suite_id": self.suite.suite_id, "task_id": spec.id,
                             "seed": self._config["seed"], "epoch": self._config["epoch"]})
        rollout = fingerprint({"group_id": group, "attempt": attempt})
        ticket = RolloutTicket(self.suite.suite_id, spec, group, rollout, attempt,
                               self._config["epoch"], int(rollout[:16], 16) % (1 << 63))
        self._cursor += 1
        return ticket

    def take(self, count: int) -> list[RolloutTicket]:
        _integer(count, "count")
        return [next(self) for _ in range(min(count, len(self)))]

    def state_dict(self) -> Json:
        return {**self._config, "cursor": self._cursor}

    def load_state_dict(self, state: Json) -> None:
        if not isinstance(state, dict) or set(state) != set(self._config) | {"cursor"}:
            raise ValueError("invalid sampler checkpoint")
        if any(state[key] != value or type(state[key]) is not type(value) for key, value in self._config.items()):
            raise ValueError("sampler checkpoint belongs to a different suite or sampling configuration")
        _integer(state["cursor"], "cursor")
        if state["cursor"] > self.total:
            raise ValueError("sampler checkpoint cursor exceeds the shard")
        self._cursor = state["cursor"]
