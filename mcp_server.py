"""
MCP server for LiveKit agent tools. Run separately: python mcp_server.py
Then point the agent at http://localhost:8000/sse (see mcp-agent.py example).
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LiveKit Agent Tools")


@mcp.tool()
def get_weather(location: str) -> str:
    """Return weather for the given location. Call when user asks about weather."""
    return f"The weather in {location} is sunny, 70°F."


# End-call is handled by the agent via livekit.agents.beta.tools.end_call.EndCallTool (in-process).
# Add more tools here as needed; they are discovered by the agent when mcp_servers is set.


if __name__ == "__main__":
    mcp.run(transport="sse")
