"""A separately installed task, with no changes to the Gym's core registry."""

from agent_gym.contracts import Task, TaskSpec, Tool, Verdict, arguments_only, number, object_schema


class Sensor:
    tools = (Tool("measure", "Apply a known temperature in degrees Celsius and read voltage in V.",
                  object_schema({"temperature_c": {"type": "number", "minimum": 0, "maximum": 100}}, ["temperature_c"])),)

    def __init__(self, coefficients):
        self.coefficients = coefficients

    def call(self, name, arguments):
        arguments_only(arguments, "temperature_c")
        temperature = number(arguments.get("temperature_c"), "temperature_c")
        if not 0 <= temperature <= 100:
            raise ValueError("temperature_c must be in [0, 100]")
        a, b, c = self.coefficients
        return {"temperature_c": temperature, "voltage_v": a + b * temperature + c * temperature**2}

    def close(self):
        pass


def make(spec: TaskSpec) -> Task:
    if spec.version != 1:
        raise ValueError("sensor-calibration supports version 1")
    rng = spec.rng()
    coefficients = (rng.uniform(-0.2, 0.2), rng.uniform(0.01, 0.03), rng.uniform(-0.00003, 0.00003))

    def verify(answer):
        try:
            if not isinstance(answer, dict):
                raise ValueError("Submit an object with a, b, c")
            a, b, c = (number(answer.get(name), name) for name in ("a", "b", "c"))
            # Reference evaluated in Horner form; truth is independent of sensor state.
            error = max(abs(a + b * t + c * t**2 - ((coefficients[2] * t + coefficients[1]) * t + coefficients[0])) for t in (17, 46, 83))
            passed = error <= 0.001
            return Verdict(passed, float(passed), "Calibration evaluated at held-out temperatures", {"maximum_error_v": error})
        except ValueError as exc:
            return Verdict(False, 0, str(exc))

    return Task(spec, "Calibrate a noiseless quadratic temperature sensor: V = a + b*T + c*T^2, with T in degrees Celsius. "
                "Measure at chosen temperatures in [0,100], then submit {a, b, c}. Units are V, V/degC, and V/degC^2. "
                "Maximum voltage prediction error at held-out temperatures must not exceed 0.001 V.", Sensor(coefficients), verify, max_actions=6)
