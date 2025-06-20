"""
MCP Client for Egile Researcher

This client can connect to the Egile Researcher MCP server and use its tools.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import structlog
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


logger = structlog.get_logger(__name__)


class EgileResearcherMCPClient:
    """
    Client for interacting with the Egile Researcher MCP server.
    """

    def __init__(self, server_command: Optional[str] = None):
        """
        Initialize the MCP client.

        Args:
            server_command: Command to start the MCP server (optional)
        """
        self.server_command = server_command or "python -m egile_researcher.server"
        self.session: Optional[ClientSession] = None
        self.available_tools: List[str] = []

    async def connect(self):
        """Connect to the MCP server."""
        try:
            # Parse the server command into command and args
            command_parts = self.server_command.split()
            command = command_parts[0]
            args = command_parts[1:] if len(command_parts) > 1 else []

            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=None,
            )

            # Connect to server
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session

                    # Initialize the session
                    await session.initialize()

                    # Get available tools
                    tools_result = await session.list_tools()
                    self.available_tools = [tool.name for tool in tools_result.tools]

                    logger.info(
                        "Connected to MCP server", available_tools=self.available_tools
                    )

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            # Session will be closed by the context manager
            self.session = None
            logger.info("Disconnected from MCP server")

    async def search_papers(
        self,
        query: str,
        days_back: Optional[int] = None,
        max_results: Optional[int] = None,
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for research papers."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        args = {"query": query}
        if days_back is not None:
            args["days_back"] = days_back
        if max_results is not None:
            args["max_results"] = max_results
        if sources is not None:
            args["sources"] = sources

        result = await self.session.call_tool("search_papers", args)
        return result.content[0].text if result.content else []

    async def summarize_paper(
        self,
        paper: Dict[str, Any],
        summary_type: str = "comprehensive",
        include_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Generate an intelligent summary of a research paper."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        args = {
            "paper": paper,
            "summary_type": summary_type,
            "include_analysis": include_analysis,
        }

        result = await self.session.call_tool("summarize_paper", args)
        return json.loads(result.content[0].text) if result.content else {}

    async def analyze_trends(
        self,
        topic: str,
        time_period: str = "last_month",
        max_papers: int = 50,
    ) -> Dict[str, Any]:
        """Analyze research trends for a specific topic."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        args = {
            "topic": topic,
            "time_period": time_period,
            "max_papers": max_papers,
        }

        result = await self.session.call_tool("analyze_trends", args)
        return json.loads(result.content[0].text) if result.content else {}

    async def compare_papers(
        self,
        papers: List[Dict[str, Any]],
        comparison_aspects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple research papers."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        args = {"papers": papers}
        if comparison_aspects is not None:
            args["comparison_aspects"] = comparison_aspects

        result = await self.session.call_tool("compare_papers", args)
        return json.loads(result.content[0].text) if result.content else {}

    async def generate_research_report(
        self,
        topic: str,
        include_trends: bool = True,
        include_summaries: bool = True,
        max_papers: int = 20,
    ) -> Dict[str, Any]:
        """Generate a comprehensive research report."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        args = {
            "topic": topic,
            "include_trends": include_trends,
            "include_summaries": include_summaries,
            "max_papers": max_papers,
        }

        result = await self.session.call_tool("generate_research_report", args)
        return json.loads(result.content[0].text) if result.content else {}

    async def search_and_summarize(
        self,
        query: str,
        max_papers: int = 10,
        summary_type: str = "brief",
        days_back: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search for papers and generate summaries in one step."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        args = {
            "query": query,
            "max_papers": max_papers,
            "summary_type": summary_type,
        }
        if days_back is not None:
            args["days_back"] = days_back

        result = await self.session.call_tool("search_and_summarize", args)
        return json.loads(result.content[0].text) if result.content else {}

    async def analyze_topic_comprehensively(
        self,
        topic: str,
        max_papers: int = 30,
        time_period: str = "last_month",
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis of a research topic."""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        args = {
            "topic": topic,
            "max_papers": max_papers,
            "time_period": time_period,
        }

        result = await self.session.call_tool("analyze_topic_comprehensively", args)
        return json.loads(result.content[0].text) if result.content else {}

    def get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        return self.available_tools.copy()


# Convenience functions for quick usage
async def quick_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Quick paper search using MCP client."""
    client = EgileResearcherMCPClient()
    try:
        await client.connect()
        return await client.search_papers(query, max_results=max_results)
    finally:
        await client.disconnect()


async def quick_analysis(topic: str) -> Dict[str, Any]:
    """Quick topic analysis using MCP client."""
    client = EgileResearcherMCPClient()
    try:
        await client.connect()
        return await client.analyze_topic_comprehensively(topic)
    finally:
        await client.disconnect()


# Example usage
async def main():
    """Example usage of the MCP client."""
    client = EgileResearcherMCPClient()

    try:
        await client.connect()
        print(f"Available tools: {client.get_available_tools()}")

        # Example: Search for papers
        papers = await client.search_papers("machine learning", max_results=5)
        print(f"Found {len(papers)} papers")

        # Example: Analyze trends
        trends = await client.analyze_trends("artificial intelligence")
        print(f"Trend analysis: {trends.get('topic')}")

    except Exception as e:
        logger.error(f"Error using MCP client: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
