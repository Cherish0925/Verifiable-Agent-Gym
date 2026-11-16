"""Small, explicit contracts. Agent observations never contain the verifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
import re
from typing import Any, Callable, Protocol

Json = dict[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    family: str
    seed: int = 0
    split: str = "train"
    difficulty: int = 1
    version: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", self.family):
            raise ValueError("family must be a lowercase task name")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if type(self.difficulty) is not int or self.difficulty not in (1, 2, 3):
            raise ValueError("difficulty must be 1, 2, or 3")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")

    @property
    def id(self) -> str:
        return f"{self.family}-v{self.version}-{self.split}-s{self.seed}-d{self.difficulty}"

    def rng(self, stream: str = "task") -> random.Random:
        return random.Random(f"{self.id}/{stream}")

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Json

    def to_dict(self) -> Json:
        return asdict(self)


class Environment(Protocol):
    tools: tuple[Tool, ...]

    def call(self, name: str, arguments: Json) -> Json:
        """ValueError means a correctable action error; other errors abort the episode."""
        ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reward: float
    reason: str
    metrics: Json = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be boolean")
        if isinstance(self.reward, bool) or not math.isfinite(self.reward) or not 0 <= self.reward <= 1:
            raise ValueError("reward must be a finite number in [0, 1]")

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass
class Task:
    spec: TaskSpec
    prompt: str
    environment: Environment
    verify: Callable[[Any], Verdict]
    max_actions: int = 32

    def observation(self) -> Json:
        return {
            "type": "start",
            "task": self.spec.to_dict(),
            "task_id": self.spec.id,
            "prompt": self.prompt,
            "tools": [tool.to_dict() for tool in self.environment.tools],
            "action_format": {"tool_call": {"tool": "name", "arguments": {}}, "final": {"submit": {}}},
            "max_actions": self.max_actions,
        }


class Agent(Protocol):
    def act(self, observation: Json, *, timeout: float) -> Json: ...
    def close(self) -> None: ...


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def arguments_only(arguments: Json, *names: str) -> None:
    extra = set(arguments) - set(names)
    if extra:
        raise ValueError(f"Unknown arguments: {', '.join(sorted(extra))}")


def object_schema(properties: Json, required: list[str] | None = None) -> Json:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}
