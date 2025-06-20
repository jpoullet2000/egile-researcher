# Egile Researcher MCP Setup

This document explains how to use the Egile Researcher as an MCP (Model Context Protocol) server and how to leverage the AI-powered agent that can intelligently use MCP tools.

## Architecture Overview

The Egile Researcher now provides multiple ways to access its capabilities:

1. **Traditional Agent** (`ResearchAgent`) - Direct Python API
2. **MCP Server** (`server.py`) - Exposes research tools via MCP protocol  
3. **AI Agent** (`AIResearchAgent`) - Intelligent agent that uses LLM reasoning to plan and execute MCP tool usage
4. **Simple Agent** (`IntelligentResearchAgent`) - Basic multi-step tool usage

## MCP Server

The MCP server exposes the following tools:

- `search_papers` - Search for research papers across multiple sources
- `summarize_paper` - Generate intelligent summaries of research papers
- `analyze_trends` - Analyze research trends for specific topics
- `generate_insights` - Generate insights from research data
- `create_bibliography` - Create formatted bibliographies

### Starting the MCP Server

```bash
# Install dependencies
poetry install

# Start the MCP server
python -m egile_researcher.server
```

### MCP Server Configuration

The server can be configured via environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"  
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment"
```

## AI-Powered Research Agent

The `AIResearchAgent` is the most intelligent way to use the research capabilities. It:

1. **Connects to MCP server** and discovers available tools
2. **Uses LLM reasoning** to create execution plans for complex tasks
3. **Executes multi-step workflows** intelligently 
4. **Enhances arguments** with context from previous steps
5. **Provides detailed results** with execution summaries

### Usage Examples

#### Simple Research Task

```python
from egile_researcher import ai_research

# Perform intelligent research with automatic planning
result = await ai_research("Find recent papers about transformer models in NLP")

print(f"Plan: {len(result['plan'])} steps")
print(f"Success: {result['summary']['successful_steps']}/{result['summary']['total_steps']}")
```

#### Advanced Usage with Manual Control

```python
from egile_researcher import AIResearchAgent

async with AIResearchAgent() as agent:
    # The agent automatically connects and discovers tools
    print(f"Available tools: {list(agent.available_tools.keys())}")
    
    # Perform intelligent research
    result = await agent.research("Research quantum computing trends and provide analysis")
    
    # Access detailed results
    for step in result['execution_results']:
        if step['success']:
            print(f"Step {step['step']}: {step['description']} ✅")
        else:
            print(f"Step {step['step']}: {step['description']} ❌ - {step['error']}")
```

#### Manual Tool Calling

```python
async with AIResearchAgent() as agent:
    # Call specific tools manually
    papers = await agent.call_tool("search_papers", {
        "query": "machine learning interpretability",
        "max_results": 5
    })
    
    # Use results in subsequent calls
    if papers:
        summary = await agent.call_tool("summarize_paper", {
            "paper": papers[0],
            "summary_type": "comprehensive"
        })
```

## How the AI Planning Works

The `AIResearchAgent` uses an LLM to intelligently plan tool usage:

1. **Task Analysis**: The LLM analyzes the user's research task
2. **Tool Selection**: Determines which tools are needed and in what order
3. **Argument Planning**: Plans specific arguments for each tool call
4. **Context Enhancement**: Uses results from previous steps to enhance subsequent calls
5. **Error Handling**: Adapts when steps fail

### Example AI Planning

For the task "Research machine learning in healthcare and analyze trends":

1. **Step 1**: Search for papers on "machine learning in healthcare" 
2. **Step 2**: Analyze trends in the topic
3. **Step 3**: Summarize key findings (if papers were found)

The agent automatically:
- Extracts relevant search terms
- Determines appropriate parameters
- Uses search results to enhance trend analysis
- Provides comprehensive execution summaries

## Simple Intelligent Agent

For simpler use cases, the `IntelligentResearchAgent` provides basic multi-step planning:

```python
from egile_researcher import quick_research

# Simple research with basic planning
result = await quick_research("quantum computing applications")
```

## Testing the MCP Setup

Run the comprehensive test suite:

```bash
# Test the AI agent
python tmp/test_ai_agent.py

# Test the MCP server directly
python tmp/test_mcp_server.py
```

## Integration with Other MCP Clients

The MCP server can be integrated with any MCP-compatible client:

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "egile-researcher": {
      "command": "python",
      "args": ["-m", "egile_researcher.server"],
      "env": {
        "AZURE_OPENAI_API_KEY": "your-api-key",
        "AZURE_OPENAI_ENDPOINT": "your-endpoint"
      }
    }
  }
}
```

### Custom MCP Client

```python
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

async with stdio_client(["python", "-m", "egile_researcher.server"]) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        
        # Call a tool
        result = await session.call_tool(CallToolRequest(
            name="search_papers",
            arguments={"query": "AI research", "max_results": 5}
        ))
```

## Benefits of the MCP Architecture

1. **Modularity**: Research tools are exposed as discrete, reusable services
2. **Interoperability**: Any MCP client can use the research capabilities  
3. **Scalability**: MCP server can handle multiple concurrent clients
4. **Intelligence**: AI agents can dynamically combine tools for complex tasks
5. **Flexibility**: Choose the right level of automation for your needs

## Troubleshooting

### Connection Issues

- Ensure the MCP server is running before starting clients
- Check that environment variables are properly set
- Verify network connectivity and ports

### Tool Execution Errors

- Check Azure OpenAI API credentials and quotas
- Verify internet connectivity for paper searches
- Review tool arguments for proper formatting

### AI Planning Issues

- The AI agent falls back to heuristic planning if LLM planning fails
- Check OpenAI API availability and token limits
- Review task descriptions for clarity
