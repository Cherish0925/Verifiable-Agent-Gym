"""Episode files are append-only; replace only small status documents."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def json_copy(value: Any) -> Any:
    return json.loads(dumps(value))


def atomic_json(path: Path, value: Any) -> None:
    text = dumps(value) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RunLease:
    """One batch writer per output directory; the OS releases it after a crash."""

    def __init__(self, directory: Path):
        self.path = directory / ".writer.lock"

    def __enter__(self):
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError(f"Another batch is writing {self.path.parent}") from exc
        return self

    def __exit__(self, *_):
        if os.name == "nt":
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()
