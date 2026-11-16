"""Synthetic evidence synthesis with revisions, exclusions, and mixed units."""

from __future__ import annotations

import math
from typing import Any

from ..contracts import Task, TaskSpec, Tool, Verdict, arguments_only, number, object_schema


class EvidenceLibrary:
    tools = (
        Tool("list_documents", "List document IDs and titles, including older revisions.", object_schema({})),
        Tool("read_document", "Read one complete document.", object_schema({"document_id": {"type": "string"}}, ["document_id"])),
        Tool("search_documents", "Find documents containing a literal phrase.", object_schema({"query": {"type": "string"}}, ["query"])),
    )

    def __init__(self, documents: dict[str, str]):
        self.documents = dict(documents)

    def call(self, name, arguments):
        if name == "list_documents":
            arguments_only(arguments)
            return {"documents": [{"id": key, "title": text.splitlines()[0].removeprefix("# ")} for key, text in self.documents.items()]}
        if name == "read_document":
            arguments_only(arguments, "document_id")
            key = arguments.get("document_id")
            if not isinstance(key, str) or key not in self.documents:
                raise ValueError("Use a document_id returned by list_documents or search_documents")
            return {"document_id": key, "text": self.documents[key], "media_type": "text/markdown"}
        arguments_only(arguments, "query")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        return {"documents": [{"id": key, "title": text.splitlines()[0]} for key, text in self.documents.items() if query.casefold() in text.casefold()]}

    def close(self):
        pass


def make(spec: TaskSpec) -> Task:
    if spec.version != 1:
        raise ValueError("evidence-aggregation supports task version 1")
    rng = spec.rng()
    documents = {}
    included = []
    values = []
    for index in range(4 + 4 * spec.difficulty):
        study = f"study-{index + 1:02}"
        revisions = 2 if spec.difficulty >= 2 and index % 3 == 0 else 1
        for revision in range(1, revisions + 1):
            # At least two eligible studies in every generated instance.
            method = "uniaxial" if index < 2 or rng.random() < 0.8 else "flexural"
            temperature = 23 if index < 2 or rng.random() < 0.75 else 60
            status = "active" if index < 2 or rng.random() < 0.8 else "retracted"
            effect_mpa = round(rng.uniform(-4, 18), 5)
            se_mpa = round(rng.uniform(0.4, 2), 5)
            unit = rng.choice(["MPa", "kPa", "Pa"]) if spec.difficulty >= 2 else "MPa"
            scale = {"MPa": 1, "kPa": 1_000, "Pa": 1_000_000}[unit]
            key = f"{study}-v{revision}"
            documents[key] = (
                f"# Tensile response: {study}, revision {revision}\n"
                f"study: {study}\nrevision: {revision}\nstatus: {status}\nmethod: {method}\n"
                f"temperature_c: {temperature}\neffect: {effect_mpa * scale:.6f}\n"
                f"standard_error: {se_mpa * scale:.6f}\nunit: {unit}\n"
                "Synthetic experiment; not a claim about a real material.\n"
            )
            if revision == revisions and status == "active" and method == "uniaxial" and 22 <= temperature <= 24:
                included.append(key)
                values.append((effect_mpa, se_mpa))
    weight = sum(1 / se**2 for _, se in values)
    expected = sum(effect / se**2 for effect, se in values) / weight
    standard_error = math.sqrt(1 / weight)
    expected_ids = frozenset(included)

    def verify(submission: Any) -> Verdict:
        try:
            if not isinstance(submission, dict):
                raise ValueError("Submission must be an object")
            mean = number(submission.get("mean_mpa"), "mean_mpa")
            se = number(submission.get("standard_error_mpa"), "standard_error_mpa")
            citations = submission.get("included_documents")
            if not isinstance(citations, list) or not all(isinstance(x, str) for x in citations):
                raise ValueError("included_documents must list document IDs")
            evidence_ok = len(citations) == len(set(citations)) and set(citations) == expected_ids
            mean_error, se_error = abs(mean - expected), abs(se - standard_error)
            passed = evidence_ok and mean_error <= 1e-5 and se_error <= 1e-5
            return Verdict(passed, float(passed), "Correct latest evidence and weighted estimate" if passed else "Evidence selection or numerical estimate differs", {
                "evidence_matches": evidence_ok, "mean_absolute_error_mpa": mean_error,
                "standard_error_absolute_error_mpa": se_error,
            })
        except ValueError as exc:
            return Verdict(False, 0, str(exc))

    prompt = (
        "Synthesize this synthetic tensile-response evidence library. For each study, use ONLY its highest revision. "
        "Include that revision only if status=active, method=uniaxial, and 22 <= temperature_c <= 24. "
        "Never fall back to an older revision of an excluded study. Convert effect and standard_error to MPa. "
        "Assume independent estimates: report the inverse-variance weighted mean and its standard error. "
        "Submit {mean_mpa: number, standard_error_mpa: number, included_documents: [document IDs]}. "
        "Absolute numerical tolerance is 0.00001 MPa. The included-document list must be exact, with no duplicates."
    )
    return Task(spec, prompt, EvidenceLibrary(documents), verify, max_actions=len(documents) + 8)
