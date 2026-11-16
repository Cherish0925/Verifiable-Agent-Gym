"""Lossless, nondispersive single-film design; normal incidence, wavelengths in nm."""

from __future__ import annotations

import cmath
import math
from typing import Any

from ..contracts import Task, TaskSpec, Tool, Verdict, arguments_only, number, object_schema


def reflectance(n: float, substrate: float, thickness: float, wavelength: float) -> float:
    r01, r12 = (1 - n) / (1 + n), (n - substrate) / (n + substrate)
    phase = cmath.exp(4j * math.pi * n * thickness / wavelength)
    return abs((r01 + r12 * phase) / (1 + r01 * r12 * phase))**2


def oracle_reflectance(n: float, substrate: float, thickness: float, wavelength: float) -> float:
    # Independent characteristic-matrix expression for the same physical model.
    delta = 2 * math.pi * n * thickness / wavelength
    b = complex(math.cos(delta), substrate / n * math.sin(delta))
    c = complex(substrate * math.cos(delta), n * math.sin(delta))
    return abs((b - c) / (b + c))**2


class ThinFilmBench:
    tools = (
        Tool("design_conditions", "Read the wavelength band, substrate, material indices, and acceptance threshold.", object_schema({})),
        Tool("simulate", "Compute normal-incidence reflectance and transmittance for one design across the target band.", object_schema({"material_id": {"type": "string"}, "thickness_nm": {"type": "number", "minimum": 20, "maximum": 400}}, ["material_id", "thickness_nm"])),
    )

    def __init__(self, conditions):
        self.conditions = conditions

    def call(self, name, arguments):
        if name == "design_conditions":
            arguments_only(arguments)
            return self.conditions
        arguments_only(arguments, "material_id", "thickness_nm")
        material_id = arguments.get("material_id")
        if not isinstance(material_id, str) or material_id not in self.conditions["materials"]:
            raise ValueError("Choose material_id from design_conditions")
        thickness = number(arguments.get("thickness_nm"), "thickness_nm")
        if not 20 <= thickness <= 400:
            raise ValueError("thickness_nm must be between 20 and 400")
        n, substrate = self.conditions["materials"][material_id], self.conditions["substrate_index"]
        samples = []
        for wavelength in self.conditions["wavelengths_nm"]:
            r = reflectance(n, substrate, thickness, wavelength)
            samples.append({"wavelength_nm": wavelength, "R": r, "T": 1 - r})
        return {"samples": samples, "mean_reflectance": sum(x["R"] for x in samples) / len(samples)}

    def close(self):
        pass


def make(spec: TaskSpec) -> Task:
    if spec.version != 1:
        raise ValueError("thin-film-design supports task version 1")
    rng = spec.rng()
    substrate, centre = rng.uniform(1.45, 2.3), rng.uniform(480, 680)
    width = (0, 0.08, 0.16)[spec.difficulty - 1]
    wavelengths = tuple(centre * (1 + width * x / 4) for x in range(-4, 5))
    indices = [math.sqrt(substrate)] + [rng.uniform(1.15, 2.4) for _ in range(2 + spec.difficulty)]
    rng.shuffle(indices)
    materials = {f"M{i + 1}": n for i, n in enumerate(indices)}
    # A feasible witness is constructed before materializing the task.
    witness_n = math.sqrt(substrate)
    witness_d = centre / (4 * witness_n)
    witness_mean = sum(oracle_reflectance(witness_n, substrate, witness_d, w) for w in wavelengths) / len(wavelengths)
    threshold = witness_mean * 1.03 + 0.00015

    def verify(submission: Any) -> Verdict:
        try:
            if not isinstance(submission, dict):
                raise ValueError("Submission must be an object")
            material = submission.get("material_id")
            if not isinstance(material, str) or material not in materials:
                raise ValueError("Unknown material_id")
            d = number(submission.get("thickness_nm"), "thickness_nm")
            if not 20 <= d <= 400:
                raise ValueError("thickness_nm outside the 20–400 nm design interval")
            mean = sum(oracle_reflectance(materials[material], substrate, d, w) for w in wavelengths) / len(wavelengths)
            passed = mean <= threshold
            return Verdict(passed, float(passed), "Design meets band-averaged reflectance target" if passed else "Band-averaged reflectance exceeds target", {"mean_reflectance": mean, "maximum_mean_reflectance": threshold})
        except ValueError as exc:
            return Verdict(False, 0, str(exc))

    conditions = {"ambient_index": 1, "substrate_index": substrate, "centre_wavelength_nm": centre,
                  "wavelengths_nm": list(wavelengths), "materials": dict(materials),
                  "thickness_range_nm": [20, 400], "maximum_mean_reflectance": threshold,
                  "assumptions": "Normal incidence; real, wavelength-independent indices; semi-infinite substrate; lossless media."}
    return Task(spec, "Design a single-layer antireflection coating using the available synthetic materials. Read design_conditions, "
                "use simulate to evaluate candidates, and submit {material_id: string, thickness_nm: number}. "
                "Any design satisfying the bounds and mean-reflectance threshold is accepted; it need not match a reference design.",
                ThinFilmBench(conditions), verify, max_actions=24)
