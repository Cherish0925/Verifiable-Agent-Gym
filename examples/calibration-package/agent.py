"""Observation-only reference agent for the calibration package, using JSONL."""

import json
import sys

voltages = []
for line in sys.stdin:
    observation = json.loads(line)
    if observation["type"] == "start":
        voltages = []
    else:
        voltages.append(observation["result"]["voltage_v"])
    if len(voltages) < 3:
        action = {"tool": "measure", "arguments": {"temperature_c": (0, 50, 100)[len(voltages)]}}
    else:
        a = voltages[0]
        c = (voltages[2] - 2 * voltages[1] + voltages[0]) / 5000
        b = (voltages[1] - a - 2500 * c) / 50
        action = {"submit": {"a": a, "b": b, "c": c}}
    print(json.dumps(action), flush=True)
