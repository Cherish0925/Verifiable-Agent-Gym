"""Run the packaged uv launcher, not an already-imported server module."""

import asyncio
import json
from pathlib import Path
import tempfile
import zipfile

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main():
    with tempfile.TemporaryDirectory(prefix="gym plugin launch ") as folder:
        workspace = Path(folder)
        with zipfile.ZipFile(next((ROOT / "dist").glob("*-plugin.zip"))) as archive:
            archive.extractall(workspace)
        package = workspace / "verifiable-agent-gym"
        manifest = json.loads((package / "mcp.json").read_text(encoding="utf-8"))
        server = manifest["mcpServers"]["agent-gym"]
        params = StdioServerParameters(command=server["command"], args=server["args"] + ["--output", str(workspace / "runs")], cwd=package / server["cwd"])
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                listed = await client.list_tools()
                if {t.name for t in listed.tools} != {"gym_tasks", "gym_open", "gym_action", "gym_close"}:
                    raise RuntimeError("Packaged server exposed unexpected tools")
                opened = await client.call_tool("gym_open", {"family": "thin-film-design"})
                if opened.isError:
                    raise RuntimeError(str(opened))
                value = opened.structuredContent or json.loads(opened.content[0].text)
                closed = await client.call_tool("gym_close", {"episode_id": value["episode_id"]})
                if closed.isError:
                    raise RuntimeError(str(closed))
    print("Relocated plugin's declared uv/MCP launcher passed.")


if __name__ == "__main__":
    asyncio.run(main())
