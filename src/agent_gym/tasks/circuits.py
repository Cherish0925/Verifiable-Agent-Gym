"""An interactive virtual instrument, not a supplied answer table."""

from __future__ import annotations

import math
import random
from typing import Any

from ..contracts import Task, TaskSpec, Tool, Verdict, arguments_only, number, object_schema


class RCInstrument:
    tools = (
        Tool("instrument_info", "Read the instrument model, units, and measurement conditions.", object_schema({})),
        Tool("measure_response", "Measure complex voltage gain at one frequency in Hz.", object_schema({"frequency_hz": {"type": "number", "minimum": 1, "maximum": 1_000_000}}, ["frequency_hz"])),
    )

    def __init__(self, gain: float, cutoff: float, noise: float, rng: random.Random):
        self.gain, self.cutoff, self.noise, self.rng = gain, cutoff, noise, rng
        self.measurements = 0

    def call(self, name, arguments):
        if name == "instrument_info":
            arguments_only(arguments)
            return {"model": "H(f) = gain / (1 + i*f/cutoff_hz)", "frequency_range_hz": [1, 1_000_000], "gain_unit": "V/V", "noise_std_relative": self.noise, "max_measurements": 8}
        arguments_only(arguments, "frequency_hz")
        frequency = number(arguments.get("frequency_hz"), "frequency_hz")
        if not 1 <= frequency <= 1_000_000:
            raise ValueError("frequency_hz must be between 1 and 1000000")
        if self.measurements >= 8:
            raise ValueError("Measurement budget exhausted; submit the fitted parameters")
        self.measurements += 1
        response = self.gain / complex(1, frequency / self.cutoff)
        sigma = abs(response) * self.noise
        response += complex(self.rng.gauss(0, sigma), self.rng.gauss(0, sigma))
        return {"frequency_hz": frequency, "real": response.real, "imag": response.imag, "unit": "V/V", "measurement": self.measurements}

    def close(self):
        pass


def make(spec: TaskSpec) -> Task:
    if spec.version != 1:
        raise ValueError("rc-identification supports task version 1")
    rng = spec.rng()
    gain, cutoff = rng.uniform(0.5, 2), 10**rng.uniform(2, 4)
    noise = (0, 0.001, 0.004)[spec.difficulty - 1]
    # The judge retains truth separately from the mutable instrument instance.
    frequencies = tuple(cutoff * factor for factor in (0.1, 0.37, 1.7, 5.3, 10))

    def verify(submission: Any) -> Verdict:
        try:
            if not isinstance(submission, dict):
                raise ValueError("Submission must be an object")
            estimate_gain = number(submission.get("gain"), "gain")
            estimate_cutoff = number(submission.get("cutoff_hz"), "cutoff_hz")
            if estimate_gain <= 0 or estimate_cutoff <= 0:
                raise ValueError("gain and cutoff_hz must be positive")
            # Polar-form oracle, distinct from the instrument's rational form.
            def response(g, c, f):
                phase = -math.atan2(f, c)
                amplitude = g / math.hypot(1, f / c)
                return complex(amplitude * math.cos(phase), amplitude * math.sin(phase))
            prediction_error = max(abs(response(estimate_gain, estimate_cutoff, f) - response(gain, cutoff, f)) / abs(response(gain, cutoff, f)) for f in frequencies)
            gain_error = abs(estimate_gain / gain - 1)
            cutoff_error = abs(estimate_cutoff / cutoff - 1)
            passed = gain_error <= 0.03 and cutoff_error <= 0.03 and prediction_error <= 0.04
            return Verdict(passed, float(passed), "Instrument identified" if passed else "Parameter or held-out response error exceeds tolerance", {
                "gain_relative_error": gain_error, "cutoff_relative_error": cutoff_error,
                "held_out_response_max_relative_error": prediction_error,
            })
        except ValueError as exc:
            return Verdict(False, 0, str(exc))

    prompt = (
        "Identify a simulated first-order RC low-pass instrument from tool measurements. Its gain lies in [0.5, 2] V/V "
        "and cutoff in [100, 10000] Hz. Select frequencies, measure, and fit. You have at most 8 measurements. "
        "Submit {gain: number, cutoff_hz: number}. Both parameter relative errors must be <= 3%, and complex-response "
        "relative error must be <= 4% at held-out frequencies. Noise conditions are available from instrument_info."
    )
    return Task(spec, prompt, RCInstrument(gain, cutoff, noise, spec.rng("measurement-noise")), verify, max_actions=16)
