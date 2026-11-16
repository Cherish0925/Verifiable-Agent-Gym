"""Build source/plugin and Skill archives from the same maintained files."""

from pathlib import Path
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_gym import __version__


def main():
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    entries = ["src", "skills", "docs", "examples", ".agents", ".claude-plugin", ".codex-plugin", ".mcp.json",
               "plugin.json", "mcp.json", "mcp.codex.json", "server.py", "pyproject.toml", "README.md", "README.zh-CN.md", "LICENSE"]
    plugin = destination / f"verifiable-agent-gym-{__version__}-plugin.zip"
    with zipfile.ZipFile(plugin, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            path = ROOT / entry
            for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if file.is_file() and not any(part == "__pycache__" or part in ("build", "dist") or part.endswith(".egg-info") for part in file.parts):
                    archive.write(file, "verifiable-agent-gym/" + file.relative_to(ROOT).as_posix())
    skill = destination / f"verifiable-agent-gym-{__version__}-skill.zip"
    with zipfile.ZipFile(skill, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        path = ROOT / "skills/verifiable-agent-gym/SKILL.md"
        archive.write(path, "verifiable-agent-gym/SKILL.md")
    print(plugin)
    print(skill)


if __name__ == "__main__":
    main()
