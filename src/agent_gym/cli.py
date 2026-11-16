from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import __version__
from .contracts import TaskSpec
from .registry import make, task_names
from .storage import dumps


def default_output() -> Path:
    configured = os.environ.get("AGENT_GYM_RUNS")
    return Path(configured) if configured else Path.home() / ".agent-gym" / "runs"


def parser():
    result = argparse.ArgumentParser(prog="agent-gym", description="Executable tasks and independent verification for tool-using agents")
    result.add_argument("--version", action="version", version=__version__)
    sub = result.add_subparsers(dest="operation", required=True)
    sub.add_parser("list", help="List installed task families")
    for name in ("run", "generate"):
        command = sub.add_parser(name)
        command.add_argument("--tasks", nargs="+", default=["all"])
        command.add_argument("--count", type=int, default=10, help="Instances per family")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--split", choices=("train", "validation", "test"), default="train")
        command.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=1)
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--agent", choices=("reference", "empty", "command"), default="reference")
            command.add_argument("--command-json", help="Agent executable and arguments as a JSON array; use absolute paths")
            command.add_argument("--jobs", type=int, default=4)
            command.add_argument("--episode-seconds", type=float, default=120)
            command.add_argument("--action-seconds", type=float, default=30)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--retry-errors", action="store_true")
            command.add_argument("--require-pass", action="store_true")
    for name in ("serve", "mcp"):
        command = sub.add_parser(name, help="Serve episodes over " + ("JSON lines" if name == "serve" else "MCP stdio"))
        command.add_argument("--output", type=Path, default=default_output())
        command.add_argument("--max-active", type=int, default=32)
        command.add_argument("--episode-seconds", type=float, default=600)
    command = sub.add_parser("export", help="Export observed actions, feedback, and rewards as JSONL")
    command.add_argument("run", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--include-unscored", action="store_true")
    command = sub.add_parser("connect", help="Write MCP or Research OS configuration for this installed interpreter")
    command.add_argument("--host", choices=("mcp", "research-os"), default="mcp")
    command.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.operation == "list":
            print("\n".join(task_names()))
        elif args.operation in ("run", "generate"):
            if args.count < 1:
                raise ValueError("count must be positive")
            families = task_names() if args.tasks == ["all"] else args.tasks
            unknown = set(families) - set(task_names())
            if unknown:
                raise ValueError(f"Unknown tasks: {', '.join(sorted(unknown))}")
            specs = [TaskSpec(family, args.seed + index, args.split, args.difficulty) for family in families for index in range(args.count)]
            if args.operation == "generate":
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                    for spec in specs:
                        task = make(spec)
                        try:
                            stream.write(dumps({"schema_version": 1, "spec": spec.to_dict(), "observation": task.observation()}) + "\n")
                        finally:
                            task.environment.close()
                print(f"Generated {len(specs)} tasks: {args.output}")
            else:
                from .process import CommandAgent
                from .reference import EmptyAgent, ReferenceAgent
                from .runner import run_batch
                config = {}
                if args.agent == "command":
                    if not args.command_json:
                        raise ValueError("--agent command requires --command-json")
                    command = json.loads(args.command_json)
                    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
                        raise ValueError("--command-json must be a nonempty JSON array of strings")
                    factory = lambda path: CommandAgent(command, path)
                    config = {"argv": command}
                else:
                    if args.command_json:
                        raise ValueError("--command-json requires --agent command")
                    factory = lambda path: ReferenceAgent() if args.agent == "reference" else EmptyAgent()
                report = run_batch(specs, args.output, factory, agent_name=args.agent, agent_config=config, jobs=args.jobs,
                                   seconds=args.episode_seconds, action_seconds=args.action_seconds, resume=args.resume, retry_errors=args.retry_errors)
                print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
                if report["interrupted"]:
                    return 130
                if report["counts"]["error"] or (args.require_pass and report["counts"]["passed"] != report["requested"]):
                    return 1
        elif args.operation in ("serve", "mcp"):
            if args.operation == "mcp":
                from .mcp_server import create_server
                create_server(args.output, max_active=args.max_active, episode_seconds=args.episode_seconds).run(transport="stdio")
            else:
                from .service import GymService, serve_jsonl
                serve_jsonl(GymService(args.output, max_active=args.max_active, episode_seconds=args.episode_seconds), sys.stdin, sys.stdout)
        elif args.operation == "export":
            from .export import export_run
            count = export_run(args.run, args.output, include_unscored=args.include_unscored)
            print(f"Exported {count} episodes: {args.output}")
        elif args.operation == "connect":
            from importlib.util import find_spec
            if find_spec("mcp") is None:
                raise RuntimeError('Install MCP support first: python -m pip install ".[mcp]"')
            server = {"command": sys.executable, "args": ["-m", "agent_gym", "mcp"]}
            if args.host == "research-os":
                server["transport"] = "stdio"
                value = {"name": "verifiable-agent-gym", "version": __version__, "description": "Executable tasks and independent verification", "mcpServers": {"agent-gym": server}}
            else:
                value = {"mcpServers": {"agent-gym": server}}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
            print(f"Wrote {args.output}. This configuration uses the installed interpreter {sys.executable}.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"agent-gym: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
