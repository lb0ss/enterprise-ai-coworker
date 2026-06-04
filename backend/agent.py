import sys
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# MCP server config — uses stdio transport, spawning mcp_server.py as a subprocess
MCP_CONFIG = {
    "docmind": {
        "command": sys.executable,
        "args": [os.path.join(os.path.dirname(__file__), "mcp_server.py")],
        "transport": "stdio",
    }
}


async def get_agent():
    """Create a ReAct agent connected to tools via the MCP protocol."""
    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()
    return create_react_agent(llm, tools)
