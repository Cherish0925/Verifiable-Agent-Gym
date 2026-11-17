---
name: verifiable-agent-gym
description: Run reproducible agent tasks, load data-backed environments, collect grouped training rollouts, or develop task/environment/verifier packages. Use for Verifiable Agent Gym, SQL task populations, scientific environments, agent evaluation and trajectory generation.
---

# Verifiable Agent Gym

## Solve a task through MCP

1. Call `gym_tasks` to discover installed task families. Call `gym_open` with a family, seed, split, and difficulty. Preserve the returned `episode_id`.
2. Read the task prompt, tool schemas, units, acceptance criteria, and remaining action count. All required observations come from this episode's tools.
3. Call `gym_action` with `episode_id` and `action: {"tool": "name", "arguments": {...}}`. Use the returned measurements or documents to decide the next action. Independent episodes can run concurrently; dependent actions within one episode stay ordered.
4. Submit with `action: {"submit": {...}}` using the task's requested fields. Report the verifier's actual result and relevant numerical metrics, not a self-assigned success score. The terminal response includes the saved result directory.
5. Use `gym_close` for an episode you abandon. A terminal episode cannot be reopened. Open a new attempt if needed; do not present repeated attempts as a first-attempt success.

`tool_error` is feedback to correct the next call. `failed` means a submitted answer did not meet the task's criteria. `error`, `truncated`, and `cancelled` are unscored; do not silently turn them into reward zero or exclude them from operational reporting.

## Evaluate or generate trajectories

From an installed package:

```sh
agent-gym run --tasks all --count 10 --difficulty 3 --jobs 4 --output runs/baseline --require-pass
agent-gym export runs/baseline --output runs/baseline.jsonl
```

The default `reference` agent is a transparent scripted baseline, not an LLM. To evaluate a model, use an external JSONL agent process with `--agent command --command-json '["/absolute/path/to/python", "/absolute/path/to/agent.py"]'`, or drive episodes directly through MCP. The process reads one observation per line and emits one action per line; diagnostics go to stderr. Use explicit executable paths on Windows as well as Unix.

Continue the same batch with `--resume`; use `--retry-errors` to retry infrastructure errors. Completed scored attempts are retained. The evaluated agent receives the task's observations and tools.

## Collect trainer-driven experience

Load finite SQL populations with `load_taskpack`, inspect `pack.describe()`, and obtain valid specifications with `pack.specs(split)`. For MCP hosts, use `agent-gym --task-pack FAMILY=PATH connect --output CONFIG`. Seeds select finite record indexes; choose from the loaded population.

Create a `TaskSuite` with attempts per task and taskpack metadata. Its rank-aware sampler keeps each task's attempts together and checkpoints issuance order. `RolloutBatch` owns independent slots; drive `reset` and `step` with partial mappings, refill completed slots, and pass a concrete `PolicyVersion`. Record actual model statistics with `policy_data` when available.

Export with `agent-gym export-rollouts RUN --output FILE`. Preserve task, source, group, policy and outcome identities when importing into Data Engineering. Read the repository's `docs/training.md` and run `examples/batched_rollouts.py` for the complete interface.

## Extend the Gym

Keep the task factory, observable tool environment, and independent verifier separate. A factory accepts `TaskSpec` and returns a `Task`; installed packages register factories through the `agent_gym.tasks` entry-point group. A solver self-report is not a verifier. State units and conventions, build a feasible task instance, and compare submitted artifacts against evidence or an independent calculation.

Reuse existing software and data through task packages. See the repository's [task-package guide](https://github.com/Cherish0925/Verifiable-Agent-Gym/blob/main/docs/task-packages.md) and [protocol](https://github.com/Cherish0925/Verifiable-Agent-Gym/blob/main/docs/protocol.md) for contracts and complete extension examples.
