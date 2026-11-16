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

The included tasks use binary rewards. Custom tasks may return graded rewards alongside an explicit pass/fail criterion. Summaries report both success among scored attempts and success among all attempted episodes; failures to run therefore remain visible.

Budgets include the final submission. A bad tool call consumes an action. In-process callbacks are cooperative: the runner checks episode deadlines between calls; long-running adapters must implement bounded execution themselves. External JSONL agents have a response timeout and an owned process group. The runner is not a container or VM boundary.

## External agent process

`--agent command` starts one process per episode in its own workspace, with explicit argv and UTF-8 pipes. It writes one observation per line to stdin. Return exactly one action per line on stdout, keeping logs on stderr. The process is closed after the terminal action. `python -m agent_gym.reference` is an executable protocol example, not a model integration.

Model SDK calls, credentials, conversation history, and native tool-schema translation belong in your agent adapter. Do not expose the server's source, verifier, or reference implementation as tools to an agent during a controlled evaluation. For stronger isolation, run the client and server on separate workers or within the isolation provided by your harness.

## JSONL service

`agent-gym serve` exposes the opposite direction: a host drives the Gym through stdin/stdout. Requests contain `id`, `method`, and `params`; responses contain the same `id` and either `result` or `error`.

```json
{"id":1,"method":"open","params":{"family":"rc-identification","seed":7}}
```

Use the returned episode ID in `action` or `close`. Supported methods are `tasks`, `open`, `action`, and `close`. This transport processes requests serially; MCP supports overlapping calls across separate episodes. Actions within one episode are serialized in both cases.

## Records and continuation

Each attempt has a generated episode ID under `<output>/<task-id>/<episode-id>/`. Its files are `episode.json`, append-only `events.jsonl`, and `result.json`; external agents additionally get a `workspace/` with stderr output. Batch-level `run.json` stores task specifications and execution configuration. `summary.json` selects one result per requested task.

Only one batch writer may use an output directory at a time. The OS releases that lease if the process crashes. `--resume` requires the same Gym version, specifications, agent configuration, and limits. Incomplete attempts restart in new environments: this is **batch continuation**, not restoration of an arbitrary solver's in-memory state.

Exports stream one episode per JSONL record and retain observations, actions, verification metrics, and reward. Unscored attempts are excluded from training export by default, with `--include-unscored` available for analysis. No model-internal reasoning, log probabilities, or imaginary interaction records are synthesized.
