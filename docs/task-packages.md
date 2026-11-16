# Task packages

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

Remote jobs need submission, bounded polling, result retrieval, and cancellation semantics in the adapter. The current core does not restore arbitrary remote jobs after a server restart. Do not claim remote continuation until a task package implements and demonstrates it.

Package optional software, data locations, and license requirements with the task package. Large datasets can stay in external storage with explicit dataset revisions; do not place credentials, proprietary datasets, or solver binaries into the core distribution. Installing a package is not a license to redistribute all of its inputs.

## Contribution scope

A useful first contribution is one end-to-end task family: a feasible instance, a reference policy using only observations, a deliberately wrong submission, and a small test that distinguishes the two. Add broader scenarios when the environment supports them. Avoid adding empty adapters, reserved directories, or duplicate tool registries as a substitute for a working integration.
