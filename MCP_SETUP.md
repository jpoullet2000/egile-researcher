# Egile Researcher MCP Server

This document explains how to use the Egile Researcher as an MCP (Model Context Protocol) server.

## Overview

The Egile Researcher has been converted from a monolithic agent architecture to an MCP server that exposes research capabilities as tools. This allows other applications (like Claude Desktop, IDEs, or custom clients) to use the research functionality through the standardized MCP protocol.

## Architecture

### Before (Monolithic Agent)
```
┌─────────────────┐
│  Application    │
├─────────────────┤
│  ResearchAgent  │
│  ├─ search()    │
│  ├─ summarize() │
│  ├─ analyze()   │
│  └─ tools/      │
└─────────────────┘
```

### After (MCP Server/Client)
```
┌─────────────────┐    MCP Protocol    ┌─────────────────┐
│  Client App     │◄──────────────────►│  MCP Server     │
│  ├─ mcp_client  │                    │  ├─ search()    │
│  └─ your_code   │                    │  ├─ summarize() │
└─────────────────┘                    │  ├─ analyze()   │
                                       │  └─ agent       │
                                       └─────────────────┘
```

## Files Created

### 1. `server.py` - MCP Server
- Exposes agent methods as MCP tools
- Uses FastMCP framework
- Handles tool registration and execution
- Manages agent lifecycle

### 2. `mcp_client.py` - MCP Client
- Python client for connecting to the MCP server
- Provides convenient methods for each tool
- Handles session management
- Includes utility functions

### 3. `test_mcp_server.py` - Test Script
- Comprehensive test suite for the MCP server
- Demonstrates all available tools
- Shows proper usage patterns

## Available MCP Tools

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `search_papers` | Search for research papers | `query`, `days_back`, `max_results`, `sources` |
| `summarize_paper` | Generate paper summaries | `paper`, `summary_type`, `include_analysis` |
| `analyze_trends` | Analyze research trends | `topic`, `time_period`, `max_papers` |
| `compare_papers` | Compare multiple papers | `papers`, `comparison_aspects` |
| `generate_research_report` | Create comprehensive reports | `topic`, `include_trends`, `include_summaries`, `max_papers` |
| `search_and_summarize` | Combined search + summarize | `query`, `max_papers`, `summary_type`, `days_back` |
| `analyze_topic_comprehensively` | Full topic analysis | `topic`, `max_papers`, `time_period` |

## Usage

### 1. Start the MCP Server

```bash
# Using the script entry point
egile-researcher-server

# Or directly
python -m egile_researcher.server
```

### 2. Use with Python Client

```python
import asyncio
from egile_researcher.mcp_client import EgileResearcherMCPClient

async def main():
    client = EgileResearcherMCPClient()
    
    try:
        await client.connect()
        
        # Search for papers
        papers = await client.search_papers("machine learning", max_results=5)
        
        # Analyze trends
        trends = await client.analyze_trends("artificial intelligence")
        
        print(f"Found {len(papers)} papers")
        print(f"Trend analysis: {trends['topic']}")
        
    finally:
        await client.disconnect()

asyncio.run(main())
```

### 3. Use with Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "egile-researcher": {
      "command": "python",
      "args": ["-m", "egile_researcher.server"],
      "env": {
        "AZURE_OPENAI_API_KEY": "your-key-here",
        "AZURE_OPENAI_ENDPOINT": "your-endpoint-here"
      }
    }
  }
}
```

### 4. Quick Testing

```bash
# Run the test suite
python tmp/test_mcp_server.py
```

## Environment Setup

Make sure you have the required environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment"
```

Or use a `.env` file:

```
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment
```

## Dependencies

The MCP setup requires these additional dependencies (already added to `pyproject.toml`):

- `fastmcp>=0.2.0` - MCP server framework
- `mcp>=1.0.0` - MCP client library

## Benefits of MCP Architecture

1. **Modularity**: Clear separation between client and server
2. **Reusability**: Multiple clients can use the same server
3. **Standardization**: Uses the standard MCP protocol
4. **Scalability**: Server can handle multiple concurrent clients
5. **Integration**: Easy integration with Claude Desktop and other MCP-compatible tools
6. **Testing**: Easier to test individual tools in isolation

## Migration Notes

The original `ResearchAgent` class is still intact and used internally by the MCP server. This means:

- ✅ All existing functionality is preserved
- ✅ No breaking changes to the core agent
- ✅ Original agent can still be used directly if needed
- ✅ Gradual migration path available

## Error Handling

The MCP server includes comprehensive error handling:

- Tool execution errors are caught and returned as MCP error responses
- Server startup/shutdown is managed gracefully
- Client connection issues are handled with proper cleanup
- Detailed logging for debugging

## Next Steps

1. **Test the MCP server**: Run `python tmp/test_mcp_server.py`
2. **Install dependencies**: `poetry install` or `pip install -e .`
3. **Configure Claude Desktop**: Add the MCP server configuration
4. **Start building**: Use the MCP tools in your applications!

## Troubleshooting

### Server Won't Start
- Check environment variables are set
- Verify dependencies are installed
- Check logs for detailed error messages

### Client Can't Connect
- Ensure server is running
- Check server command path
- Verify MCP protocol compatibility

### Tool Execution Fails
- Check Azure OpenAI configuration
- Verify API quotas and limits
- Review tool parameters and types
