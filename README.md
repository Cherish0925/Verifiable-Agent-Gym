# Verifiable Agent Gym

**Executable research tasks, tool environments, and independent verification for agents.**

[中文](README.zh-CN.md) · [Install in an agent host](docs/install.md) · [Protocol](docs/protocol.md) · [Build a task package](docs/task-packages.md)

An agent should learn from what it actually does. This Gym gives it a task, tools that act on an environment, and a verifier that evaluates the submitted result. The same core works through Python, a command-line runner, JSON lines, or MCP. It does not require a particular model, cloud account, or agent framework.

The project is public and MIT-licensed. Use it as a standalone evaluation environment, install it as an agent plugin, or contribute domain-specific task packages. It is not an LLM provider, a replacement research OS, or an RL optimizer.

## Run it

Python 3.11 or newer. Core tasks use only the standard library; MCP is an optional extra.

```sh
git clone https://github.com/Cherish0925/Verifiable-Agent-Gym.git
cd Verifiable-Agent-Gym
python -m pip install -e .
agent-gym list
agent-gym run --tasks all --count 10 --difficulty 3 --jobs 4 --output runs/baseline --require-pass
agent-gym export runs/baseline --output runs/baseline.jsonl
```

On a system without a `python` command, use `python3` on macOS/Linux or `py -3.11` on Windows. `python -m agent_gym` also works wherever the console entry point is not on PATH. No Bash, WSL, Rust compiler, or model API key is required.

The default agent is a **scripted reference policy**, not an LLM. It solves through the same observations and tools exposed to other agents. An `empty` policy supplies a negative control. To evaluate your model, connect it through MCP or an external JSONL process.

## Use it from an agent

```sh
python -m pip install -e ".[mcp]"
agent-gym connect --output agent-gym.mcp.json
```

Import that configuration into a compatible MCP host. The generated command uses the exact Python interpreter where the Gym was installed. For Codex and Claude Code, this repository also includes native plugin manifests and a shared [Skill](skills/verifiable-agent-gym/SKILL.md). [Installation details](docs/install.md) include Research OS configuration and plugin launch requirements.

Four MCP tools cover the lifecycle:

| Tool | Operation |
| --- | --- |
| `gym_tasks` | Discover installed task families. |
| `gym_open` | Create an episode with a specified seed, split, and difficulty. |
| `gym_action` | Call an environment tool or submit the final result. |
| `gym_close` | Close an unfinished episode. |

For example, open `rc-identification`, call `instrument_info` and `measure_response` through `gym_action`, then submit the fitted gain and cutoff. Each episode owns its state. Several episodes may run concurrently without mixing measurements or outputs.

## Included task families

These are generated, executable reference environments—not a claim to cover all of science.

| Family | Agent's work | Independent acceptance test |
| --- | --- | --- |
| `evidence-aggregation` | Read a synthetic document library, choose current eligible revisions, convert units, combine measurements. | Exact evidence selection and inverse-variance estimate, calculated from the generated source records. |
| `rc-identification` | Choose measurement frequencies and identify a noisy virtual instrument. | Parameter errors and complex-response error at held-out frequencies, using a separate analytical expression. |
| `thin-film-design` | Explore available materials and film thicknesses for an antireflection coating. | A characteristic-matrix calculation checks a band-averaged target; any qualifying design is accepted. |

Every family supports deterministic seeds, three difficulties, and separate train/validation/test seed namespaces. The optical generator constructs a feasible design before issuing a task. Measurement noise has its own reproducible random stream.

## Evaluation that remains interpretable

- **A wrong answer is not a broken environment.** Verified submissions become `passed` or `failed`, with a reward. Infrastructure errors, time/action exhaustion, and cancellation remain unscored and visible in the summary.
- **Evidence survives the process.** Every attempt records its task, ordered observations/actions, terminal result, and numerical verifier metrics. Export emits observed trajectories and rewards, not fabricated reasoning or token probabilities.
- **Resume does not rewrite history.** `--resume` skips finished attempts. Interrupted attempts restart in fresh environments; `--retry-errors` explicitly retries infrastructure failures. Old attempts remain available.
- **Concurrency has an owner.** The batch runner admits at most `--jobs` episodes at once. External agents get separate subprocesses and working directories; one episode's actions remain ordered.

For external agents, pass an executable and arguments, not a shell string:

```sh
agent-gym run --tasks rc-identification --count 20 --agent command --command-json '["/absolute/path/to/python", "/absolute/path/to/my_agent.py"]' --jobs 4 --output runs/model
```

The external process reads one JSON observation from stdin and returns one JSON action on stdout. [Protocol examples and Python usage](docs/protocol.md) show the complete exchange.

## Grow through task packages

Datasets, knowledge bases, simulators, and laboratory software should extend the environment rather than become mandatory core dependencies. A package registers task factories in the `agent_gym.tasks` entry-point group. Each factory provides:

1. A task specification and observable instructions.
2. An executable environment with typed tools and per-episode state.
3. A verifier that checks results against evidence, physical constraints, reference calculations, or executable tests.

[The complete calibration example](examples/calibration-package/) is independently installable and demonstrates this extension point. The same contract can wrap an existing solver or remote instrument client. Skills explain how to work; MCP carries calls; plugins package discovery and installation. None replaces the environment or verifier.

## Scope and next work

The current runner is local and single-machine. Built-in environments are lightweight; external agent processes have response deadlines. In-process task providers must bound their own long-running tool calls. This is execution ownership, **not an OS sandbox** for untrusted code.

Public task generators and reference policies are useful for development, not secret benchmark answers. Seed splits do not establish semantic out-of-distribution generalization. Open-ended research without a defensible verifier must not be relabeled as verified success.

Next integration work is concrete: task packages for IOP/optical solvers, durable remote jobs, and trainer-specific rollout adapters. Those are not represented as working connectors in this release. The JSONL export is a foundation for training data; tokenization, policy log probabilities, and optimization remain the trainer's responsibility.

## Develop

```sh
python -m pip install -e ".[mcp]"
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
python scripts/package_plugin.py
```

CI runs the episode, subprocess, MCP, and installation paths on Linux, Windows, and macOS. Contributions should add a working capability or resolve a demonstrated failure. For a task package, include a feasible example, a wrong-answer case, explicit units/assumptions, and a verifier independent of the agent's self-report. Keep optional software optional.

Conceptual references: [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), [Gymnasium's environment interface](https://gymnasium.farama.org/api/env/), and [Model Context Protocol](https://modelcontextprotocol.io/). This project is independent of them and does not claim API compatibility with Gymnasium or a specific RL trainer.
