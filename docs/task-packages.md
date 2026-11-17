# Task packages

A domain can supply task records or a Python task factory. Records are useful when many instances share one environment profile; factories add new environment behavior.

## Data-backed SQL tasks

```sh
agent-gym pack-info examples/sql-workload/tasks.jsonl --family warehouse-sql
agent-gym --task-pack warehouse-sql=examples/sql-workload/tasks.jsonl list
agent-gym --task-pack warehouse-sql=examples/sql-workload/tasks.jsonl mcp
```

`--task-pack` is a global option placed before the subcommand. `connect` retains its absolute path in the generated host configuration. Each pack is a finite population: seeds select sorted record IDs within a split. `generate` and `run` select all matching records by default; `--seed` chooses the starting index, `--difficulty` filters records and `--count` limits the selection. Generated task families keep the defaults of 10 instances at difficulty 1. In Python, `pack.specs(split)` returns valid specifications with their recorded difficulties.

```python
from agent_gym.taskpacks import load_taskpack

pack = load_taskpack("task-release", family="warehouse-sql")
specs = pack.specs("train")
task = pack.make(specs[0])
```

The loader accepts canonical record JSONL, task JSONL, or a Data Engineering source-release directory. A canonical task record has `kind: "task"`, source identity, a fixed split, and this payload:

```json
{
  "profile": "sql/v1",
  "instruction": "Return each department and its total budget, ordered by department.",
  "database": {
    "ddl": ["CREATE TABLE projects (department TEXT, budget REAL)"],
    "tables": {"projects": {"columns": ["department", "budget"], "rows": [["optics", 12], ["optics", 8], ["software", 5]]}}
  },
  "verifier": {"sql": "SELECT department, SUM(budget) FROM projects GROUP BY department ORDER BY department", "ordered": true},
  "difficulty": 1,
  "max_actions": 32
}
```

The database can instead be `{"path":"warehouse.sqlite","sha256":"<file-hash>"}`, resolved against the pack's asset root. A verifier can supply expected `rows` instead of reference `sql`; it also supports `columns`, `atol` and `rtol`.

Each episode has a database snapshot and exposes `database_schema` and `query`. Submit `{"submit":{"sql":"..."}}` or result `rows`. Equivalent SQL is accepted by comparing results, preserving duplicate multiplicity and optional ordering. Reward is matched-row F1; a complete match passes. Reference queries and results remain outside observations. Dataset IDs, source revisions and pack/release hashes are retained in the task's `data_reference`.

The four authored [SQL examples](../examples/sql-workload/) exercise joins, grouping, subqueries and held-out queries on one database. Run `python examples/sql-workload/agent.py --output runs/sql-example` for a multi-step policy, or use the [training interface](training.md) for repeated grouped attempts.

## Python task factories

A task package is an ordinary Python distribution. It can contain datasets and knowledge resources or wrap existing databases, simulators, MCP services, and remote jobs. The Gym does not need to install every solver to discover a package's task names.

Register a factory in the package's `pyproject.toml`:

```toml
[project.entry-points."agent_gym.tasks"]
sensor-calibration = "calibration_gym:make"
```

The factory accepts a `TaskSpec` and returns a `Task` with the same specification, a prompt, an environment, a verifier, and an action budget. The environment provides `tools`, `call(name, arguments)`, and `close()`. A verifier returns `Verdict(passed, reward, reason, metrics)`.

See [the complete calibration package](../examples/calibration-package/) for executable source, packaging metadata, and a reference policy. Install it into the same interpreter as the Gym. It then appears in `agent-gym list` and `gym_tasks` without editing the registry.

## What belongs in a task

- **A solvable objective.** Generate parameters with a feasible witness, preserve an executable reference procedure, or use curated instances with known acceptance conditions. An arbitrary prompt plus a confidence score is not a verified task.
- **A real environment.** Tool calls must read or change concrete state. Give each episode its own mutable workspace or remote job; shared resources should be immutable or have an explicit concurrency policy.
- **An independent acceptance test.** Use evidence, analytical predictions, tests, or a separate numerical method. A solver reporting “success” does not prove the submitted result is correct.
- **Enough scientific context.** State units, coordinate/sign conventions, tolerances, approximations, and the observable conditions the agent needs. Put them in task/tool output, not only in developer documentation.

A task revision identifies changes to generation, observation, or acceptance semantics. Reject unsupported revisions. Seeded random streams should separate task generation from stochastic measurements. Keep train/validation/test namespaces separate; if the research requires transfer across domains or methods, define those splits explicitly rather than relying on seed changes.

## Connect existing research environments

For IOP, an optical solver, or a database, a task adapter should translate Gym tools into the existing system's calls. It should preserve returned artifact paths, physical metadata, and job IDs. The verifier reads actual resulting artifacts or repeats an independent calculation. Closing the episode must release only resources owned by that episode.

Remote adapters own job submission, polling, result retrieval, cancellation and any continuation state.

Declare optional software and data locations with the task package. Keep large datasets in external storage with explicit revisions.

## Contribution scope

A useful first contribution is one end-to-end task family: a feasible instance, a reference policy using only observations, a deliberately wrong submission, and a small test that distinguishes the two. Add broader scenarios when the environment supports them. Avoid adding empty adapters, reserved directories, or duplicate tool registries as a substitute for a working integration.
