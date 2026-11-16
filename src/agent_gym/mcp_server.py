"""MCP is an optional transport, not a second implementation of the Gym."""

from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
from typing import Any

from .service import GymService


def create_server(output: Path, *, max_active: int = 32, episode_seconds: float = 600):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError('Install MCP support with: python -m pip install ".[mcp]"') from exc
    service = GymService(output, max_active=max_active, episode_seconds=episode_seconds)

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield service
        finally:
            await asyncio.to_thread(service.close_all)

    server = FastMCP("Verifiable Agent Gym", lifespan=lifespan, log_level="WARNING", instructions=(
        "List task families, open an episode, act through its tools, then submit a result. "
        "A gym_action tool call is {tool: name, arguments: {...}}; a final action is {submit: {...}}. "
        "The returned task prompt defines the acceptance criteria. Close episodes you abandon."
    ))

    @server.tool()
    async def gym_tasks() -> dict[str, Any]:
        """List available task families, dataset splits, and difficulty levels."""
        return await asyncio.to_thread(service.tasks)

    @server.tool()
    async def gym_open(family: str, seed: int = 0, split: str = "train", difficulty: int = 1, version: int = 1) -> dict[str, Any]:
        """Create an independent episode and return its task, tools, and action budget."""
        return await asyncio.to_thread(service.open, family, seed, split, difficulty, version)

    @server.tool()
    async def gym_action(episode_id: str, action: dict[str, Any]) -> dict[str, Any]:
        """Call one environment tool or submit the final answer for independent verification."""
        return await asyncio.to_thread(service.action, episode_id, action)

    @server.tool()
    async def gym_close(episode_id: str) -> dict[str, Any]:
        """Close an unfinished episode without assigning a false failure reward."""
        return await asyncio.to_thread(service.close, episode_id)

    return server
