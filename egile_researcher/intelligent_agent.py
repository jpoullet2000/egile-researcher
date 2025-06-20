"""
Intelligent Agent that uses MCP tools dynamically.

This agent can discover MCP tools and use them intelligently to accomplish complex research tasks.
"""

from typing import Any, Dict, List, Optional

import structlog
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolRequest, Tool

from .config import ResearchAgentConfig, AzureOpenAIConfig


logger = structlog.get_logger(__name__)


class IntelligentResearchAgent:
    """
    An intelligent agent that can use MCP tools dynamically to accomplish research tasks.
    """

    def __init__(
        self,
        config: Optional[ResearchAgentConfig] = None,
        server_command: Optional[str] = None,
    ):
        """
        Initialize the intelligent agent.

        Args:
            config: Research agent configuration
            server_command: Command to start the MCP server
        """
        self.config = config or ResearchAgentConfig(
            openai_config=AzureOpenAIConfig.from_environment()
        )
        self.server_command = server_command or "python -m egile_researcher.server"
        self.session: Optional[ClientSession] = None
        self.available_tools: Dict[str, Tool] = {}

    async def connect(self):
        """Connect to the MCP server and discover available tools."""
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

                    # Discover available tools
                    tools_response = await session.list_tools()
                    self.available_tools = {
                        tool.name: tool for tool in tools_response.tools
                    }

                    logger.info(
                        f"Connected to MCP server with {len(self.available_tools)} tools"
                    )
                    for tool_name in self.available_tools.keys():
                        logger.info(f"Available tool: {tool_name}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from MCP server")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a specific MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not available")

        try:
            request = CallToolRequest(name=tool_name, arguments=arguments)
            response = await self.session.call_tool(request)

            logger.info(f"Called tool '{tool_name}' with arguments: {arguments}")
            return response.content

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise

    def get_available_tools(self) -> Dict[str, str]:
        """
        Get information about available tools.

        Returns:
            Dictionary mapping tool names to their descriptions
        """
        return {
            name: tool.description or "No description available"
            for name, tool in self.available_tools.items()
        }

    async def plan_and_execute(self, task: str) -> Dict[str, Any]:
        """
        Plan and execute a complex research task using available MCP tools.

        This method uses AI to reason about which tools to use and in what order.

        Args:
            task: Description of the research task to accomplish

        Returns:
            Results of the task execution
        """
        if not self.session:
            await self.connect()

        # Get tool descriptions for planning
        tool_descriptions = self.get_available_tools()

        # Create a plan using AI reasoning
        plan = await self._create_plan(task, tool_descriptions)

        # Execute the plan
        results = await self._execute_plan(plan)

        return {
            "task": task,
            "plan": plan,
            "results": results,
        }

    async def _create_plan(
        self, task: str, tool_descriptions: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Create an execution plan for the given task.

        Args:
            task: The task to plan for
            tool_descriptions: Available tools and their descriptions

        Returns:
            List of planned steps
        """
        # This is a simplified planning logic. In a real implementation,
        # you would use an LLM to generate a proper plan.

        plan = []

        # Basic heuristics for common research tasks
        if "search" in task.lower() or "find" in task.lower():
            plan.append(
                {
                    "step": 1,
                    "tool": "search_papers",
                    "description": "Search for relevant papers",
                    "arguments": {"query": task},
                }
            )

        if "summarize" in task.lower() or "summary" in task.lower():
            plan.append(
                {
                    "step": len(plan) + 1,
                    "tool": "summarize_paper",
                    "description": "Summarize the papers",
                    "arguments": {"summary_type": "comprehensive"},
                }
            )

        if "trend" in task.lower() or "analysis" in task.lower():
            plan.append(
                {
                    "step": len(plan) + 1,
                    "tool": "analyze_trends",
                    "description": "Analyze trends in the topic",
                    "arguments": {"topic": task},
                }
            )

        # Default: search for papers if no specific plan
        if not plan:
            plan.append(
                {
                    "step": 1,
                    "tool": "search_papers",
                    "description": "Search for relevant papers",
                    "arguments": {"query": task},
                }
            )

        logger.info(f"Created plan with {len(plan)} steps for task: {task}")
        return plan

    async def _execute_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the planned steps.

        Args:
            plan: List of planned steps

        Returns:
            Results from each step
        """
        results = []
        previous_results = None

        for step in plan:
            step_num = step["step"]
            tool_name = step["tool"]
            description = step["description"]
            arguments = step.get("arguments", {})

            logger.info(f"Executing step {step_num}: {description}")

            try:
                # Modify arguments based on previous results if needed
                if previous_results and "paper" in arguments:
                    # If we need a paper and have previous search results
                    if isinstance(previous_results, list) and len(previous_results) > 0:
                        arguments["paper"] = previous_results[0]

                result = await self.call_tool(tool_name, arguments)

                step_result = {
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "arguments": arguments,
                    "result": result,
                    "success": True,
                }

                results.append(step_result)
                previous_results = result

                logger.info(f"Step {step_num} completed successfully")

            except Exception as e:
                step_result = {
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "arguments": arguments,
                    "error": str(e),
                    "success": False,
                }

                results.append(step_result)
                logger.error(f"Step {step_num} failed: {e}")

        return results

    async def research(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a research task with the given query.

        Args:
            query: Research query
            **kwargs: Additional parameters

        Returns:
            Research results
        """
        task = f"Research: {query}"
        return await self.plan_and_execute(task)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Convenience function for quick research tasks
async def quick_research(query: str, **kwargs) -> Dict[str, Any]:
    """
    Perform a quick research task.

    Args:
        query: Research query
        **kwargs: Additional parameters

    Returns:
        Research results
    """
    async with IntelligentResearchAgent() as agent:
        return await agent.research(query, **kwargs)
