"""
AI-Powered Research Agent that uses LLM reasoning to plan and execute MCP tool usage.

This agent uses an LLM to intelligently plan which tools to use and in what sequence
to accomplish complex research tasks.
"""

from typing import Any, Dict, List, Optional

import structlog
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool

from .config import ResearchAgentConfig, AzureOpenAIConfig
from .client import AzureOpenAIClient


logger = structlog.get_logger(__name__)


class AIResearchAgent:
    """
    An AI-powered research agent that uses LLM reasoning to plan and execute MCP tool usage.
    """

    def __init__(
        self,
        config: Optional[ResearchAgentConfig] = None,
        server_command: Optional[str] = None,
    ):
        """
        Initialize the AI research agent.

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

        # Initialize OpenAI client using the same approach as the main agent
        self.openai_client = AzureOpenAIClient(self.config.openai_config)

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
            response = await self.session.call_tool(tool_name, arguments)

            logger.info(f"Called tool '{tool_name}' with arguments: {arguments}")
            return response.content

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}': {e}")
            raise

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the LLM prompt."""
        tool_descriptions = []
        for name, tool in self.available_tools.items():
            description = tool.description or "No description available"

            # Format input schema if available
            input_schema = ""
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                properties = tool.inputSchema.get("properties", {})
                if properties:
                    params = []
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        params.append(f"  - {param_name} ({param_type}): {param_desc}")
                    input_schema = "\n  Parameters:\n" + "\n".join(params)

            tool_descriptions.append(f"- {name}: {description}{input_schema}")

        return "\n".join(tool_descriptions)

    async def _create_ai_plan(self, task: str) -> List[Dict[str, Any]]:
        """
        Use LLM to create an intelligent execution plan.

        Args:
            task: The research task to plan for

        Returns:
            List of planned steps with tool calls
        """
        tools_info = self._format_tools_for_prompt()

        system_prompt = f"""You are an intelligent research planning assistant. Your job is to create a step-by-step plan to accomplish research tasks using available MCP tools.

Available tools:
{tools_info}

Rules:
1. Create a logical sequence of tool calls to accomplish the task
2. Each step should have: step_number, tool_name, description, and arguments
3. Consider dependencies between steps (e.g., you need to search papers before summarizing them)
4. Be specific with arguments - extract relevant information from the task
5. If the task is complex, break it into multiple steps
6. Output valid JSON format

Example output format:
[
  {{
    "step": 1,
    "tool": "search_papers",
    "description": "Search for relevant papers on the topic",
    "arguments": {{"query": "machine learning in healthcare", "max_results": 10}}
  }},
  {{
    "step": 2,
    "tool": "analyze_trends",
    "description": "Analyze trends in the research area",
    "arguments": {{"topic": "machine learning in healthcare", "time_period": "last_month"}}
  }}
]"""

        user_prompt = f"""Create a step-by-step plan to accomplish this research task: "{task}"

Consider what information the user might want and plan accordingly. Think about:
- What papers or information to search for
- Whether summaries are needed
- If trend analysis would be valuable
- The logical order of operations

Respond with a JSON array of steps."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.openai_client.chat_completion(
                messages=messages,
                model=self.config.openai_config.default_model,
                temperature=0.3,
                max_tokens=1000,
            )

            plan_text = response.choices[0].message.content.strip()

            # Parse the JSON response
            import json

            plan = json.loads(plan_text)

            logger.info(f"AI created plan with {len(plan)} steps for task: {task}")
            return plan

        except Exception as e:
            logger.error(f"Failed to create AI plan: {e}")
            # Fallback to simple heuristic planning
            return await self._fallback_plan(task)

    async def _fallback_plan(self, task: str) -> List[Dict[str, Any]]:
        """
        Fallback planning using simple heuristics.

        Args:
            task: The task to plan for

        Returns:
            Simple plan based on heuristics
        """
        plan = []

        # Extract query from task
        query = task.lower()

        # Always start with searching
        plan.append(
            {
                "step": 1,
                "tool": "search_papers",
                "description": "Search for relevant papers",
                "arguments": {"query": task, "max_results": 10},
            }
        )

        # Add trend analysis if requested
        if "trend" in query or "analysis" in query:
            plan.append(
                {
                    "step": 2,
                    "tool": "analyze_trends",
                    "description": "Analyze trends in the research area",
                    "arguments": {"topic": task, "time_period": "last_month"},
                }
            )

        logger.info(f"Created fallback plan with {len(plan)} steps")
        return plan

    async def _execute_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the planned steps intelligently.

        Args:
            plan: List of planned steps

        Returns:
            Results from each step
        """
        results = []
        context = {}  # Shared context between steps

        for step in plan:
            step_num = step["step"]
            tool_name = step["tool"]
            description = step["description"]
            arguments = step.get("arguments", {})

            logger.info(f"Executing step {step_num}: {description}")

            try:
                # Enhance arguments with context from previous steps
                enhanced_arguments = await self._enhance_arguments_with_context(
                    tool_name, arguments, context
                )

                result = await self.call_tool(tool_name, enhanced_arguments)

                # Store useful information in context for next steps
                context[f"step_{step_num}_result"] = result
                if tool_name == "search_papers" and isinstance(result, list):
                    context["papers"] = result

                step_result = {
                    "step": step_num,
                    "tool": tool_name,
                    "description": description,
                    "arguments": enhanced_arguments,
                    "result": result,
                    "success": True,
                }

                results.append(step_result)
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

    async def _enhance_arguments_with_context(
        self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance tool arguments with context from previous steps.

        Args:
            tool_name: Name of the tool being called
            arguments: Original arguments
            context: Context from previous steps

        Returns:
            Enhanced arguments
        """
        enhanced = arguments.copy()

        # If we need a paper and have papers from search
        if tool_name == "summarize_paper" and "paper" not in enhanced:
            papers = context.get("papers", [])
            if papers:
                enhanced["paper"] = papers[0]  # Use first paper
                logger.info("Enhanced summarize_paper with paper from search results")

        return enhanced

    async def research(self, task: str) -> Dict[str, Any]:
        """
        Perform an intelligent research task.

        Args:
            task: Description of the research task

        Returns:
            Complete research results with plan and execution details
        """
        if not self.session:
            await self.connect()

        # Create AI-powered plan
        plan = await self._create_ai_plan(task)

        # Execute the plan
        results = await self._execute_plan(plan)

        return {
            "task": task,
            "plan": plan,
            "execution_results": results,
            "summary": self._create_summary(results),
        }

    def _create_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of the execution results."""
        successful_steps = [r for r in results if r.get("success", False)]
        failed_steps = [r for r in results if not r.get("success", False)]

        return {
            "total_steps": len(results),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "tools_used": list(set(r["tool"] for r in successful_steps)),
            "has_errors": len(failed_steps) > 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # ClientSession will be closed automatically by the context manager
        self.session = None


# Convenience function for quick AI-powered research
async def ai_research(task: str) -> Dict[str, Any]:
    """
    Perform AI-powered research with automatic tool selection and planning.

    Args:
        task: Description of what you want to research

    Returns:
        Complete research results
    """
    async with AIResearchAgent() as agent:
        return await agent.research(task)
