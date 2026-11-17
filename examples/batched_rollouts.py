"""Run grouped policy attempts; replace ReferenceAgent with a batched model call."""

import argparse
from pathlib import Path

from agent_gym.contracts import TaskSpec
from agent_gym.reference import ReferenceAgent
from agent_gym.rollout import PolicyVersion, RolloutBatch, export_rollouts
from agent_gym.suite import TaskSuite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = TaskSuite([TaskSpec("rc-identification", seed=i) for i in range(4)],
                      attempts_per_task=4, name="rc-grouped-reference")
    sampler = suite.sampler(seed=17)
    policies = {}
    rows = {}
    with RolloutBatch(args.output, slots=4, policy=PolicyVersion("scripted-reference", "1")) as batch:
        while len(sampler) or batch.active_slots:
            free = [slot for slot in range(4) if slot not in batch.active_slots]
            tickets = sampler.take(len(free))
            opened = batch.reset(dict(zip(free, tickets)))
            rows.update(opened)
            for slot in opened:
                policies[slot] = ReferenceAgent()
            # An external trainer can batch its own model generation here and
            # use row["rollout"]["sample_seed"] for each independent attempt.
            actions = {slot: policies[slot].act(row["observation"], timeout=30)
                       for slot, row in rows.items() if not row["done"]}
            stepped = batch.step(actions)
            for slot, row in stepped.items():
                if row["done"]:
                    policies.pop(slot).close()
                    rows.pop(slot)
                else:
                    rows[slot] = row
        print(batch.summary())
    exported = export_rollouts(args.output, args.output / "trajectories.jsonl")
    print(f"Exported {exported} factual episodes from {suite.suite_id}")


if __name__ == "__main__":
    main()
