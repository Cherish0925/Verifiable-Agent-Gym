# An independently installed task package

From the main repository root, after installing the Gym:

```sh
python -m pip install ./examples/calibration-package
agent-gym list
```

`sensor-calibration` now appears through the `agent_gym.tasks` entry point. The task gives an agent a virtual temperature sensor and accepts coefficients only if they predict voltages at held-out temperatures within 1 mV. The reference policy obtains its data through measurements, not the verifier.

Use `gym_open` with this family from MCP, or run the included JSONL agent with explicit executable paths:

```sh
agent-gym run --tasks sensor-calibration --agent command --command-json '["/absolute/path/to/python", "/absolute/path/to/Verifiable-Agent-Gym/examples/calibration-package/agent.py"]' --count 10 --jobs 4 --output runs/calibration --require-pass
```

The core `reference` policy covers only the three built-in families. Use this package's policy or your own agent for this family. The example deliberately has no extra scientific dependencies; a production instrument adapter can implement the same contracts around its own client library.
