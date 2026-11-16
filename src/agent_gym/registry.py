"""Lazy task-pack discovery. Installing a pack does not execute it during listing."""

from importlib.metadata import entry_points
from typing import Callable

from .contracts import Task, TaskSpec

_factories: dict[str, Callable[[TaskSpec], Task]] = {}
_builtins = ("evidence-aggregation", "rc-identification", "thin-film-design")


def register(name: str, factory: Callable[[TaskSpec], Task]) -> None:
    TaskSpec(name)
    if name in task_names():
        raise ValueError(f"Task family already registered: {name}")
    _factories[name] = factory


def task_names() -> list[str]:
    return sorted(set(_builtins) | set(_factories) | {e.name for e in entry_points(group="agent_gym.tasks")})


def make(spec: TaskSpec) -> Task:
    if spec.family in _builtins:
        from .tasks import evidence, optics, circuits
        factory = {"evidence-aggregation": evidence.make, "thin-film-design": optics.make, "rc-identification": circuits.make}[spec.family]
    elif spec.family in _factories:
        factory = _factories[spec.family]
    else:
        matches = [e for e in entry_points(group="agent_gym.tasks") if e.name == spec.family]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous task {spec.family}; available: {', '.join(task_names())}")
        factory = matches[0].load()
    task = factory(spec)
    if task.spec != spec:
        task.environment.close()
        raise ValueError("Task factory returned a different specification")
    return task
