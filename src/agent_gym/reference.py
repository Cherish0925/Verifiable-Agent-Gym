"""Transparent baselines that solve from observations, never from a Task or judge."""

from __future__ import annotations

import json
import math
import sys


class ReferenceAgent:
    def __init__(self):
        self.family = ""
        self.documents = []
        self.records = []
        self.measurements = []
        self.candidates = []
        self.design_results = []

    def act(self, observation, *, timeout=30):
        if observation["type"] == "tool_error":
            raise RuntimeError(observation["error"])
        if observation["type"] == "start":
            self.family = observation["task"]["family"]
            first = {"evidence-aggregation": "list_documents", "rc-identification": "instrument_info", "thin-film-design": "design_conditions"}
            if self.family not in first:
                raise ValueError(f"No reference policy for {self.family}")
            return {"tool": first[self.family], "arguments": {}}
        result, tool = observation["result"], observation["tool"]
        if self.family == "evidence-aggregation":
            if tool == "list_documents":
                self.documents = [d["id"] for d in result["documents"]]
            elif tool == "read_document":
                record = {key: value for key, value in (line.split(": ", 1) for line in result["text"].splitlines() if ": " in line and not line.startswith("#"))}
                record["document_id"] = result["document_id"]
                self.records.append(record)
            if self.documents:
                return {"tool": "read_document", "arguments": {"document_id": self.documents.pop(0)}}
            latest = {}
            for record in self.records:
                study = record["study"]
                if study not in latest or int(record["revision"]) > int(latest[study]["revision"]):
                    latest[study] = record
            selected = [r for r in latest.values() if r["status"] == "active" and r["method"] == "uniaxial" and 22 <= float(r["temperature_c"]) <= 24]
            values = [(float(r["effect"]) * {"MPa": 1, "kPa": 1e-3, "Pa": 1e-6}[r["unit"]], float(r["standard_error"]) * {"MPa": 1, "kPa": 1e-3, "Pa": 1e-6}[r["unit"]]) for r in selected]
            weights = [1 / se**2 for _, se in values]
            total = math.fsum(weights)
            return {"submit": {"mean_mpa": math.fsum(w * x[0] for w, x in zip(weights, values)) / total, "standard_error_mpa": 1 / math.sqrt(total), "included_documents": [r["document_id"] for r in selected]}}
        if self.family == "rc-identification":
            if tool == "instrument_info":
                return {"tool": "measure_response", "arguments": {"frequency_hz": 1000}}
            real, imag, frequency = result["real"], result["imag"], result["frequency_hz"]
            cutoff = -frequency * real / imag
            gain = (real**2 + imag**2) / real
            self.measurements.append((gain, cutoff))
            if len(self.measurements) < 5:
                return {"tool": "measure_response", "arguments": {"frequency_hz": min(10000, max(100, cutoff))}}
            # The initial pilot measurement locates the informative frequency band.
            values = self.measurements[1:]
            return {"submit": {"gain": math.fsum(x[0] for x in values) / len(values), "cutoff_hz": math.fsum(x[1] for x in values) / len(values)}}
        if tool == "design_conditions":
            centre = result["centre_wavelength_nm"]
            self.candidates = [{"material_id": key, "thickness_nm": centre / (4 * n)} for key, n in result["materials"].items()]
        elif tool == "simulate":
            self.design_results.append((result["mean_reflectance"], self.current_design))
        if self.candidates:
            self.current_design = self.candidates.pop(0)
            return {"tool": "simulate", "arguments": self.current_design}
        return {"submit": min(self.design_results, key=lambda item: item[0])[1]}

    def close(self):
        pass


class EmptyAgent:
    """Negative control: a well-formed protocol message with an incorrect answer."""

    def act(self, observation, *, timeout=30):
        return {"submit": {}}

    def close(self):
        pass


def main():
    policy = ReferenceAgent()
    for line in sys.stdin:
        observation = json.loads(line)
        if observation.get("type") == "start":
            policy = ReferenceAgent()
        print(json.dumps(policy.act(observation), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
