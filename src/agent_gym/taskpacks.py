"""Load finite, versioned SQL task populations from records or data releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import TaskSpec
from .tasks.sql import make_sql_task


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=pairs)
    _canonical(value)
    return value


def _file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _record(value, forced_split=None):
    if not isinstance(value, dict):
        raise ValueError("task record must be an object")
    if value.get("schema_version") == "1.0":
        if value.get("kind") != "task":
            return None
        payload = value.get("payload")
    elif type(value.get("schema_version")) is int and value["schema_version"] == 1:
        payload = value.get("task")
    else:
        raise ValueError("task records require canonical schema 1.0 or task schema 1")
    identifier, source = value.get("id"), value.get("source")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("task record requires a nonempty id")
    if not isinstance(source, dict) or any(not isinstance(source.get(k), str) or not source[k].strip() for k in ("uri", "revision")):
        raise ValueError("task source requires uri and revision")
    split = value.get("split") or forced_split
    if split not in ("train", "validation", "test") or (forced_split is not None and split != forced_split):
        raise ValueError("task record must preserve its release split")
    if not isinstance(payload, dict) or payload.get("profile") != "sql/v1":
        raise ValueError("task payload must use profile sql/v1")
    difficulty = payload.get("difficulty", 1)
    if type(difficulty) is not int or difficulty not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2 or 3")
    return {"id": identifier, "source": source, "split": split, "payload": payload,
            "difficulty": difficulty, "allowed_uses": value.get("allowed_uses", [])}


class SQLTaskPack:
    def __init__(self, records: list[dict], *, family: str, version: int, asset_root: Path,
                 pack_sha256: str, release_id: str | None):
        TaskSpec(family, version=version)
        self.family, self.version = family, version
        self.asset_root = asset_root.resolve()
        self.pack_sha256, self.release_id = pack_sha256, release_id
        self._by_split = {split: sorted((r for r in records if r["split"] == split), key=lambda r: r["id"])
                          for split in ("train", "validation", "test")}

    def describe(self) -> dict:
        return {"family": self.family, "version": self.version, "profile": "sql/v1",
                "pack_sha256": self.pack_sha256, "release_id": self.release_id,
                "splits": {split: len(rows) for split, rows in self._by_split.items()}}

    def specs(self, split: str | None = None) -> list[TaskSpec]:
        if split is not None and split not in self._by_split:
            raise ValueError("unknown task split")
        return [TaskSpec(self.family, index, name, row["difficulty"], self.version)
                for name, rows in self._by_split.items() if split is None or split == name
                for index, row in enumerate(rows)]

    def make(self, spec: TaskSpec):
        if spec.family != self.family or spec.version != self.version:
            raise ValueError("task family/version does not match the loaded pack")
        rows = self._by_split[spec.split]
        if spec.seed >= len(rows):
            raise ValueError("seed is outside this finite task population")
        record = rows[spec.seed]
        if spec.difficulty != record["difficulty"]:
            raise ValueError("task difficulty does not match its record")
        reference = {"record_id": record["id"], "source": record["source"], "pack_sha256": self.pack_sha256,
                     "release_id": self.release_id, "profile": "sql/v1", "split": record["split"],
                     "allowed_uses": record["allowed_uses"]}
        return make_sql_task(spec, record["payload"], reference, self.asset_root)

    def register(self):
        from .registry import register
        register(self.family, self.make)
        return self


def load_taskpack(path: str | Path, *, family: str = "sql-dataset", version: int = 1,
                  asset_root: str | Path | None = None) -> SQLTaskPack:
    path = Path(path).resolve(strict=True)
    release_id = None
    if path.is_dir():
        manifest = _read_json((path / "manifest.json").read_text(encoding="utf-8"))
        release_id = manifest.get("release_id")
        if manifest.get("schema_version") != "1.0" or _digest({k: v for k, v in manifest.items() if k != "release_id"}) != release_id:
            raise ValueError("invalid Data Engineering release manifest")
        files = []
        for split in ("train", "validation", "test"):
            filename = split + ".jsonl"
            file = path / filename
            info = manifest["files"].get(filename)
            if not isinstance(info, dict) or _file_hash(file) != info.get("sha256") or file.stat().st_size != info.get("bytes"):
                raise ValueError(f"release partition hash mismatch: {filename}")
            files.append((file, split))
        default_root = path
    elif path.is_file():
        files, default_root = [(path, None)], path.parent
    else:
        raise ValueError("task pack must be JSONL or a release directory")
    records, identifiers = [], set()
    for file, forced_split in files:
        before = _file_hash(file)
        with file.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = _record(_read_json(line), forced_split)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{file.name}:{line_number}: {exc}") from exc
                if record is None:
                    continue
                if record["id"] in identifiers:
                    raise ValueError(f"duplicate task record id: {record['id']}")
                identifiers.add(record["id"])
                records.append(record)
        if _file_hash(file) != before:
            raise ValueError("task input changed while reading")
    if not records:
        raise ValueError("task pack contains no SQL tasks")
    # The digest follows task content and identities, not path or line order.
    fingerprint = _digest(sorted(records, key=lambda r: r["id"]))
    return SQLTaskPack(records, family=family, version=version,
                       asset_root=Path(asset_root) if asset_root is not None else default_root,
                       pack_sha256=fingerprint, release_id=release_id)
