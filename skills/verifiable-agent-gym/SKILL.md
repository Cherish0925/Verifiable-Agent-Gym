---
name: verifiable-agent-gym
description: Run tool-using research tasks with independent verification, evaluate an agent across reproducible task instances, or develop a task/environment/verifier package. Use for Verifiable Agent Gym, virtual instrument identification, evidence aggregation, optical design exercises, and verified trajectory generation.
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

Continue the same batch with `--resume`. Add `--retry-errors` only when deliberately retrying infrastructure errors. Completed scored attempts are retained, not overwritten. For a benchmark, keep the verifier and reference implementation outside the evaluated agent's available tools and context; the published source is for development, not hidden-test secrecy.

## Extend the Gym

Keep the task factory, observable tool environment, and independent verifier separate. A factory accepts `TaskSpec` and returns a `Task`; installed packages register factories through the `agent_gym.tasks` entry-point group. A solver self-report is not a verifier. State units and conventions, build a feasible task instance, and compare submitted artifacts against evidence or an independent calculation.

Reuse existing software and data through task packages; do not duplicate solver runtimes in the Gym core. Dataset licenses, external executables, and model credentials belong to their respective integrations. See `docs/task-packages.md` and `docs/protocol.md` in the repository for contracts and a complete extension example.
