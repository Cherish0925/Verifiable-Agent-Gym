# Execution protocol

The task factory owns generation and verification. An episode owns a fresh environment and ordered actions. The agent sees observations and tool schemas, not the verifier callable. Transports reuse those objects rather than implementing another agent loop.

## Python

```python
from pathlib import Path
from agent_gym.contracts import TaskSpec
from agent_gym.episode import Episode
from agent_gym.registry import make

with Episode(make(TaskSpec("rc-identification", seed=7)), Path("runs")) as episode:
    print(episode.initial)
    feedback = episode.step({"tool": "measure_response", "arguments": {"frequency_hz": 1000}})
    # Supply your own policy here; final parameters are shown only as an example.
    result = episode.step({"submit": {"gain": 1.2, "cutoff_hz": 800}})
    print(result)
```

A built-in environment exposes JSON-schema tool descriptions and accepts JSON objects. `ValueError` from `Environment.call` is a correctable action error. Other exceptions, invalid environment output, and verifier exceptions end the episode as infrastructure errors. Verification is final: gather evidence before submitting.

## Action and observation

One action is exactly one of:

```json
{"tool": "measure_response", "arguments": {"frequency_hz": 1000}}
```

```json
{"submit": {"gain": 1.2, "cutoff_hz": 800}}
```

The first observation has `type: "start"`, task identity, instructions, tools, action format, and budget. Subsequent observations have `type: "tool_result"` or `"tool_error"`, with `remaining_actions`. A terminal response has `type: "terminal"` and `result`.

| Result status | Meaning | Reward |
| --- | --- | --- |
| `passed` | A submission satisfies the verifier. | Assigned by verifier, in [0, 1]. |
| `failed` | A submission does not satisfy the verifier. | Assigned by verifier, in [0, 1]. |
| `truncated` | Action or time budget exhausted before a valid conclusion. | `null` |
| `error` | Setup, environment, verifier, agent transport, or cleanup failed. | `null` |
| `cancelled` | Caller or service stopped the episode. | `null` |

The scientific reference tasks use binary rewards; SQL tasks use matched-row F1 and pass on the complete expected result. Custom tasks define their reward and pass criterion. Summaries report success among scored attempts and among all attempted episodes.

Budgets include the final submission. A bad tool call consumes an action. The runner checks episode deadlines between calls; long-running adapters implement their own operation deadlines. External JSONL agents have a response timeout and an owned process group.

## External agent process

`--agent command` starts one process per episode in its own workspace, with explicit argv and UTF-8 pipes. It writes one observation per line to stdin. Return exactly one action per line on stdout, keeping logs on stderr. The process is closed after the terminal action. `python -m agent_gym.reference` is an executable protocol example, not a model integration.

Model SDK calls, conversation history and native tool-schema translation belong in the agent adapter. The evaluation interface supplies observations and tools; reference solutions stay with the task implementation.

## JSONL service

`agent-gym serve` exposes the opposite direction: a host drives the Gym through stdin/stdout. Requests contain `id`, `method`, and `params`; responses contain the same `id` and either `result` or `error`.

```json
{"id":1,"method":"open","params":{"family":"rc-identification","seed":7}}
```

Use the returned episode ID in `action` or `close`. Supported methods are `tasks`, `open`, `action`, and `close`. This transport processes requests serially; MCP supports overlapping calls across separate episodes. Actions within one episode are serialized in both cases.

## Records and continuation

Each attempt has a generated episode ID under `<output>/<task-id>/<episode-id>/`. Its files are `episode.json`, append-only `events.jsonl`, and `result.json`; external agents additionally get a `workspace/` with stderr output. Batch-level `run.json` stores task specifications and execution configuration. `summary.json` selects one result per requested task.

Only one batch writer may use an output directory at a time. The OS releases that lease if the process crashes. `--resume` requires the same Gym version, specifications, agent configuration, and limits. Incomplete attempts restart in new environments: this is **batch continuation**, not restoration of an arbitrary solver's in-memory state.

`export` streams one episode per JSONL record with observations, actions, metrics and reward; use `--include-unscored` to include attempts without a score. For trainer-driven batches, [TaskSuite and RolloutBatch](training.md) preserve task groups, policy revisions and supplied model statistics. Their `export-rollouts` command includes unscored episodes by default, with `--scored-only` available for selection.
