# Verifiable Agent Gym

Build reusable task distributions, executable environments and feedback for learning agents. The core serves external models and trainers through task packages, batched interactions and recorded outcomes.

Start from what the agent observes, what it can change, how an episode develops and what constitutes a successful result. Read `docs/protocol.md` for execution semantics and `docs/task-packages.md` before adding a domain.

- Make a task reusable across instances and policies. Keep task data, environment behavior and result evaluation separately understandable.
- Reuse Episode for ordered interaction and terminal results. Preserve group, attempt, policy and source identities through exports.
- Treat observation shape, action budgets, resource configuration and terminal status as part of the experiment.
- Add meaningful wrong-answer, alternative-solution and environment-failure cases. Measure throughput when changing batch execution.
- Keep models, trainers and domain dependencies in their adapters. Share one implementation across Python, CLI, JSONL and MCP.
- Keep explanations focused on capabilities and operating semantics. Avoid repeated caveats and competing architecture documents.
- Use topic branches and pull requests; retain unrelated changes.

Run `python -m unittest discover -s tests -v`, the affected task/rollout example, and packaging checks when installation changes.
