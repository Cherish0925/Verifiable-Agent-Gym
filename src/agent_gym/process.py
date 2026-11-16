"""JSON-line agent transport; explicit argv, UTF-8, and owned process lifetime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from typing import Any

from .storage import dumps


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


class CommandAgent:
    """One trusted local process per episode. This is not an OS security sandbox."""

    def __init__(self, command: list[str], workspace: Path):
        if not command or not all(isinstance(arg, str) and arg for arg in command):
            raise ValueError("command must be a nonempty argv list")
        workspace.mkdir(parents=True, exist_ok=True)
        self.stderr = (workspace / "agent.stderr.log").open("w", encoding="utf-8")
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        try:
            self.process = subprocess.Popen(
                command, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                text=True, encoding="utf-8", errors="strict", bufsize=1, shell=False,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                env=environment,
            )
        except Exception:
            self.stderr.close()
            raise
        self.messages: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.outgoing: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.reader.start()
        self.writer.start()

    def _write(self):
        while not self.closed.is_set():
            try:
                line, completed = self.outgoing.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except Exception as exc:
                completed.put(exc)
            else:
                completed.put(None)

    def _read(self):
        try:
            while not self.closed.is_set():
                line = self.process.stdout.readline(1_048_577)
                if not line:
                    raise RuntimeError("Agent exited before returning an action; see agent.stderr.log")
                if len(line) > 1_048_576:
                    raise RuntimeError("Agent action exceeds the 1 MiB protocol limit")
                value = json.loads(line, parse_constant=reject_constant)
                while not self.closed.is_set():
                    try:
                        self.messages.put(value, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            while not self.closed.is_set():
                try:
                    self.messages.put(exc, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def act(self, observation, *, timeout=30):
        if self.closed.is_set():
            raise RuntimeError("Agent process is closed")
        deadline = time.monotonic() + timeout
        completed = queue.Queue(maxsize=1)
        try:
            self.outgoing.put((dumps(observation) + "\n", completed), timeout=timeout)
            error = completed.get(timeout=max(0, deadline - time.monotonic()))
            if error is not None:
                raise error
            response = self.messages.get(timeout=max(0, deadline - time.monotonic()))
        except (queue.Empty, queue.Full) as exc:
            self.close()
            raise TimeoutError("Agent response deadline exceeded") from exc
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        process = self.process
        if os.name == "nt":
            if process.poll() is None:
                taskkill = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"
                try:
                    subprocess.run([str(taskkill), "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
        else:
            # The session belongs to this process; do not affect other agents.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        self.reader.join(timeout=1)
        self.writer.join(timeout=1)
        process.stdin.close()
        process.stdout.close()
        self.stderr.close()
