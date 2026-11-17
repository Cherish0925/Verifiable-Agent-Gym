# Verifiable Agent Gym

**Task distributions, executable environments and observed feedback for learning agents.**

[中文](README.zh-CN.md) · [Training interface](docs/training.md) · [Protocol](docs/protocol.md) · [Task packages](docs/task-packages.md) · [Agent installation](docs/install.md)

An agent receives a task, acts through environment tools and submits a result for independent evaluation. This project makes that experience reusable across policies: task suites define what to attempt, batched slots execute the interactions, and exports retain the attempts and outcomes that an external learner consumes.

The model-neutral core serves evaluation, synthetic experience collection, post-training and scientific experimentation. [Data Engineering](https://github.com/Cherish0925/Data-Engineering) supplies versioned task and trajectory products; Gym supplies executable environments; the trainer owns policy inference and optimization. [Public frontier workloads](https://github.com/Cherish0925/Data-Engineering/blob/main/docs/frontier-workloads.md) motivate this division.

## Run an evaluation or collect grouped rollouts

Python 3.11 or newer. The core uses the standard library and is MIT-licensed.

```sh
git clone https://github.com/Cherish0925/Verifiable-Agent-Gym.git
cd Verifiable-Agent-Gym
python -m pip install -e .
agent-gym list
agent-gym run --tasks all --count 10 --difficulty 3 --jobs 4 --output runs/baseline --require-pass
agent-gym export runs/baseline --output runs/baseline.jsonl
python examples/batched_rollouts.py --output runs/grouped
```

The evaluation command and grouped example use a scripted reference policy. The grouped example collects four attempts for each of four RC identification tasks through four episode slots, then writes `runs/grouped/trajectories.jsonl`. Replace its action-generation step with your model's batched inference. The `empty` policy provides an unsuccessful comparison for the evaluation runner.

`python -m agent_gym` is equivalent to `agent-gym`. Use `python3` or `py -3.11` where appropriate.

## Task suites and the learner interface

| Object | Responsibility |
| --- | --- |
| `TaskSuite` | Finite, versioned task definitions and attempts per task; a content-derived suite identity. |
| `SuiteSampler` | Deterministic task ordering, epoch and sample seeds, whole-task group assignment across ranks and a restorable issuance cursor. |
| `PolicyVersion` | Policy name, revision and generation settings attached to each attempt. |
| `RolloutBatch` | Concurrent environment slots with selective `reset`, `step` and `close_slots`; completed slots can be refilled independently. |
| Rollout export | Task, group, attempt and policy identities; ordered interactions, supplied policy-side statistics, terminal status and verifier results. |

```python
from agent_gym import TaskSpec
from agent_gym.suite import TaskSuite

suite = TaskSuite(
    [TaskSpec("rc-identification", seed=i) for i in range(100)],
    attempts_per_task=4,
    name="rc-policy-comparison",
)
sampler = suite.sampler(seed=17, rank=0, world_size=4)
tickets = sampler.take(8)
checkpoint = sampler.state_dict()
```

All attempts for one sampled task group stay on the same rank. A batch returns `terminated`, `truncated`, `scored`, `reward` and the observation for each stepped slot. Policy-provided token statistics can accompany actions; the trainer uses those statistics to construct its loss. The sampler checkpoint restores which tickets are issued next. [Training integration](docs/training.md) defines the complete loop and the boundary between sampling state and live environment state.

```sh
agent-gym export-rollouts runs/grouped --output runs/grouped-export.jsonl
```

## Dataset-backed tasks

A task pack makes a dataset executable. The SQL profile accepts a database snapshot, an instruction and a reference query or result. The agent inspects the schema, queries the database and submits SQL or rows. The verifier compares results with explicit row-order and numerical-tolerance semantics, accepting alternative queries with the same result.

```sh
agent-gym pack-info examples/sql-workload/tasks.jsonl --family warehouse-sql
agent-gym --task-pack warehouse-sql=examples/sql-workload/tasks.jsonl list
python examples/sql-workload/agent.py --output runs/sql-example
```

The example contains four authored relational questions with fixed train/validation/test assignments. `load_taskpack` accepts their JSONL file or a Data Engineering release, preserves source identities and returns the finite task specifications in each split. After installing Data Engineering, the same records can be prepared and loaded as a release:

```sh
ai-data build examples/sql-workload/data.recipe.json --output runs/sql-tasks
agent-gym pack-info runs/sql-tasks --family warehouse-sql
```

Use `pack.specs("train")` to construct a `TaskSuite` and `pack.make` as the batch's task factory. [Task package documentation](docs/task-packages.md) covers custom providers; [the calibration package](examples/calibration-package/) demonstrates the `agent_gym.tasks` entry point.

Data Engineering's [experience-cycle example](https://github.com/Cherish0925/Data-Engineering/blob/main/examples/experience_cycle.py) runs the complete path: task release, eight grouped SQL attempts, trajectory release, training shards and resumable rank reads.

## Generated scientific environments

| Family | Agent's work | Evaluation |
| --- | --- | --- |
| `evidence-aggregation` | Select eligible document revisions, convert units and combine measurements. | Evidence membership, inverse-variance estimate and standard error. |
| `rc-identification` | Choose frequencies, measure a noisy instrument and identify its parameters. | Parameter error and response at held-out frequencies, calculated independently. |
| `thin-film-design` | Choose materials and thicknesses for an antireflection coating. | A characteristic-matrix calculation checks the band-averaged target. |

These families generate repeatable instances with three difficulties and separate split seed namespaces. SQL packs instead enumerate dataset records. Transfer experiments can hold out task mechanisms, domains or compositions in addition to instances.

## Connect an agent

```sh
python -m pip install -e ".[mcp]"
agent-gym connect --output agent-gym.mcp.json
```

A compatible MCP host uses `gym_tasks`, `gym_open`, `gym_action` and `gym_close`. Codex and Claude Code plugin manifests and a shared [Skill](skills/verifiable-agent-gym/SKILL.md) are included; see [installation](docs/install.md).

An external process can read JSON observations from stdin and write JSON actions to stdout:

```sh
agent-gym run --tasks rc-identification --count 20 --agent command --command-json '["/absolute/path/to/python", "/absolute/path/to/my_agent.py"]' --jobs 4 --output runs/model
```

The [protocol](docs/protocol.md) defines this exchange. Evaluation runs retain every attempt; `--resume` skips completed attempts and starts interrupted work in a fresh environment. `--retry-errors` retries recorded infrastructure failures. Verified answers have rewards; interruption, truncation and environment failures retain their status and an unscored reward.

## Current execution scope

Execution is local: batch slots run environment calls in threads, and the command runner owns external agent processes. Long-running task providers manage their own solver jobs. Live environment restoration, remote workers and framework-specific training adapters remain integration work. Tokenization, model probabilities and optimization belong to the external learner. The included fixtures and policies exercise these interfaces; model-learning and large-workload results require consumer experiments.

```sh
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
python scripts/package_plugin.py
```

Read [AGENTS.md](AGENTS.md) before contributing. A useful addition supplies a working environment, an interoperable learner adapter or a research result that changes how tasks are constructed and sampled.
