import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from agent_gym.reference import ReferenceAgent


@unittest.skipUnless(importlib.util.find_spec("mcp"), "install the mcp extra for transport tests")
class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_discovery_parallel_episodes_and_verification(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory(prefix="gym mcp ") as folder:
            params = StdioServerParameters(command=sys.executable, args=["-m", "agent_gym", "mcp", "--output", folder])
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    self.assertEqual({t.name for t in tools.tools}, {"gym_tasks", "gym_open", "gym_action", "gym_close"})

                    async def call(name, arguments):
                        result = await client.call_tool(name, arguments)
                        self.assertFalse(result.isError, result)
                        return result.structuredContent or json.loads(result.content[0].text)

                    families = (await call("gym_tasks", {}))["families"]
                    async def solve(family):
                        observation = await call("gym_open", {"family": family, "seed": 17, "difficulty": 3})
                        episode_id, policy = observation["episode_id"], ReferenceAgent()
                        while observation["type"] != "terminal":
                            action = policy.act(observation)
                            observation = await call("gym_action", {"episode_id": episode_id, "action": action})
                        self.assertEqual(observation["result"]["status"], "passed")
                        self.assertTrue(Path(observation["result_directory"]).is_dir())
                    await asyncio.gather(*(solve(family) for family in families))
                    abandoned = await call("gym_open", {"family": families[0]})
                    closed = await call("gym_close", {"episode_id": abandoned["episode_id"]})
                    self.assertEqual(closed["status"], "cancelled")
                    self.assertIsNone(closed["reward"])
