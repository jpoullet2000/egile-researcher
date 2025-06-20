# Egile Researcher

🔬 **AI-Powered Research Automation with MCP Integration**

Egile Researcher is an intelligent research assistant that helps researchers, academics, and knowledge workers stay up-to-date with the latest publications in their field. It provides automated summarization, analysis, and insights from recent academic papers and research publications.

**🆕 Now featuring:**
- **MCP Server Architecture** - Expose research tools as standardized services
- **AI-Powered Planning** - Intelligent agents that plan and execute multi-step research workflows
- **Dynamic Tool Discovery** - Agents that adapt to available research capabilities
- **Multi-Client Support** - Integration with Claude Desktop, custom applications, and more

## 🏗️ Architecture Options

Egile Researcher offers multiple ways to access research capabilities:

### 1. 🧠 AI Research Agent (Recommended)
Intelligent agent that uses LLM reasoning to plan and execute complex research tasks:

```python
from egile_researcher import ai_research

# AI automatically plans and executes multi-step research
result = await ai_research("Find recent papers about quantum machine learning and analyze trends")
```

### 2. 🔧 Traditional Research Agent
Direct Python API for programmatic access:

```python
from egile_researcher import ResearchAgent

agent = ResearchAgent()
papers = await agent.search_papers("machine learning")
```

### 3. 🌐 MCP Server
Standardized server exposing research tools for any MCP-compatible client:

```bash
# Start MCP server
python -m egile_researcher.server
```

### 4. 🔌 MCP Client Integration
Use with Claude Desktop, VS Code, or custom applications.

## 🚀 Features

### 🧠 AI-Powered Intelligence
- **Intelligent Planning**: AI agents that create optimal execution plans for complex research tasks
- **Dynamic Tool Discovery**: Agents automatically discover and use available research tools
- **Multi-Step Reasoning**: Chain research operations intelligently based on context
- **Fallback Mechanisms**: Graceful degradation when AI services are unavailable

### 🔬 Core Research Capabilities
- **Publication Discovery**: Find recent papers from arXiv, PubMed, Google Scholar, and other sources
- **Intelligent Summarization**: AI-powered summaries extracting key findings and methodologies
- **Research Analysis**: Deep analysis of research trends, citation patterns, and emerging topics
- **Comparative Studies**: Compare multiple papers and identify relationships between research works
- **Citation Tracking**: Monitor citations and impact of specific papers or research areas
- **Topic Clustering**: Group related publications and identify research themes

### 🌐 MCP Integration
- **MCP Server**: Expose research capabilities as standardized MCP tools
- **Claude Desktop Integration**: Use research tools directly in Claude Desktop
- **Multi-Client Support**: Compatible with any MCP client application
- **Tool Interoperability**: Research tools can be combined with other MCP services

### 🛠️ Available MCP Tools
- `search_papers` - Search for research papers across multiple sources
- `summarize_paper` - Generate intelligent summaries of research papers  
- `analyze_trends` - Analyze research trends for specific topics
- `compare_papers` - Compare multiple research papers
- `generate_research_report` - Generate comprehensive research reports
- `search_and_summarize` - Search for papers and generate summaries in one step
- `analyze_topic_comprehensively` - Perform comprehensive analysis of research topics

## 📦 Installation

### Using Poetry (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd egile-researcher

# Install with Poetry
poetry install

# Activate the environment
poetry shell
```

### Using pip
```bash
pip install egile-researcher
```

## 🛠️ Configuration

### Environment Variables
Create a `.env` file with your configuration:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEFAULT_MODEL=gpt-4.1-mini
AZURE_USE_MANAGED_IDENTITY=false

# Research Database APIs (optional)
ARXIV_API_KEY=your_arxiv_key
PUBMED_API_KEY=your_pubmed_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key
```

## 📚 Quick Start

### 🧠 AI Research Agent (Simplest)

The AI Research Agent is the easiest way to get started. It automatically plans and executes complex research tasks:

```python
import asyncio
from egile_researcher import ai_research

async def main():
    # AI automatically plans and executes research workflow
    result = await ai_research("Find recent papers about transformer models in NLP")
    
    # View the execution plan
    print(f"AI created {len(result['plan'])} step plan:")
    for step in result['plan']:
        print(f"  {step['step']}: {step['description']}")
    
    # Check results
    summary = result['summary']
    print(f"✅ {summary['successful_steps']}/{summary['total_steps']} steps completed")

if __name__ == "__main__":
    asyncio.run(main())
```

### 🤖 Advanced AI Agent Usage

For more control over the AI agent:

```python
from egile_researcher import AIResearchAgent

async def advanced_research():
    async with AIResearchAgent() as agent:
        # Agent automatically discovers available tools
        print(f"Available tools: {list(agent.available_tools.keys())}")
        
        # Perform intelligent research
        result = await agent.research("Research quantum computing trends and provide analysis")
        
        # Access detailed execution results
        for step in result['execution_results']:
            if step['success']:
                print(f"✅ Step {step['step']}: {step['description']}")
            else:
                print(f"❌ Step {step['step']}: {step['error']}")
        
        # Or call specific tools manually
        papers = await agent.call_tool("search_papers", {
            "query": "machine learning interpretability",
            "max_results": 5
        })

asyncio.run(advanced_research())
```

### 🔧 Traditional Agent Usage

Direct API access for programmatic control:

```python
import asyncio
from egile_researcher import ResearchAgent
from egile_researcher.config import ResearchAgentConfig, AzureOpenAIConfig

async def traditional_usage():
    # Configure the research agent
    openai_config = AzureOpenAIConfig.from_environment()
    agent_config = ResearchAgentConfig(
        openai_config=openai_config,
        research_areas=["machine learning", "artificial intelligence"],
        max_papers_per_search=10
    )
    
    # Initialize the agent
    agent = ResearchAgent(agent_config)
    
    # Search for recent papers
    papers = await agent.search_papers(
        query="large language models",
        days_back=7,
        max_results=5
    )
    
    # Summarize papers
    for paper in papers:
        summary = await agent.summarize_paper(paper)
        print(f"Title: {paper['title']}")
        print(f"Summary: {summary['summary']}")
        print(f"Key Findings: {summary['key_findings']}")
        print("-" * 50)
    
    await agent.close()

asyncio.run(traditional_usage())
```

### 🌐 MCP Server Usage

#### Starting the MCP Server

```bash
# Start the MCP server
python -m egile_researcher.server
```

#### Claude Desktop Integration

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

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

#### Custom MCP Client

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

async def use_mcp_client():
    # Configure server parameters with proper command splitting
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "egile_researcher.server"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            # Call a research tool (using correct format)
            result = await session.call_tool(
                "search_papers",
                {"query": "AI research", "max_results": 5}
            )
            print(f"Search results: {result.content}")

asyncio.run(use_mcp_client())
```

### 🎯 Advanced Research Workflows

```python
# AI agent can handle complex multi-step research tasks
result = await ai_research("""
    Research the latest developments in quantum machine learning, 
    analyze trends over the past year, and provide insights on 
    future research directions
""")

# Traditional agent for specific operations
trends = await agent.analyze_trends(
    topic="transformer architectures",
    time_period="last_month"
)

comparison = await agent.compare_papers([paper1, paper2, paper3])

report = await agent.generate_research_report(
    topic="recent advances in NLP",
    include_visualizations=True
)
```

## 🏗️ Architecture

### Core Components

- **AI Research Agent** (`AIResearchAgent`): Intelligent agent using LLM reasoning for research planning
- **Traditional Research Agent** (`ResearchAgent`): Direct API for programmatic research tasks
- **MCP Server** (`server.py`): Exposes research capabilities as MCP tools
- **MCP Client** (`mcp_client.py`): Python client for MCP server integration
- **Research Tools** (`tools/`): Individual research tool implementations
- **Configuration System** (`config.py`): Type-safe configuration management

### AI Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AI Research Agent                       │
├─────────────────────────────────────────────────────────────┤
│  🧠 LLM Planning Engine                                     │
│  ├─ Task Analysis                                          │
│  ├─ Tool Selection                                         │
│  ├─ Multi-step Planning                                    │
│  └─ Context Enhancement                                    │
├─────────────────────────────────────────────────────────────┤
│  🔌 MCP Client Interface                                   │
│  ├─ Dynamic Tool Discovery                                 │
│  ├─ Tool Execution                                         │
│  └─ Session Management                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ MCP Protocol
┌─────────────────────────────────────────────────────────────┐
│                      MCP Server                            │
├─────────────────────────────────────────────────────────────┤
│  🛠️ Research Tools                                         │
│  ├─ search_papers                                          │
│  ├─ summarize_paper                                        │
│  ├─ analyze_trends                                         │
│  ├─ generate_insights                                      │
│  └─ create_bibliography                                    │
├─────────────────────────────────────────────────────────────┤
│  🔬 Research Agent Core                                    │
│  ├─ Publication Fetchers                                   │
│  ├─ Content Processors                                     │
│  ├─ AI Summarizers                                         │
│  └─ Trend Analyzers                                        │
└─────────────────────────────────────────────────────────────┘
```

### MCP Integration Benefits

1. **Modularity**: Research tools as discrete, reusable services
2. **Interoperability**: Compatible with any MCP client
3. **Scalability**: Server handles multiple concurrent clients
4. **Intelligence**: AI agents can dynamically combine tools
5. **Flexibility**: Choose the right level of automation for your needs

## 🧪 Testing & Examples

### Run the AI Agent Demo

```bash
# Test AI-powered planning capabilities
python tmp/demo_planning.py

# Test basic AI agent functionality  
python tmp/test_basic.py

# Test AI agent with MCP server integration
python tmp/test_ai_agent.py
```

### Example AI Planning Output

```
🤖 AI Research Agent Planning Demo
==================================================
✅ AI Agent created

🧠 Testing AI-powered planning...
✅ AI plan created with 3 steps:
  • Step 1: Search for recent papers about transformer models published in the last year
  • Step 2: Generate summaries for the found papers to provide concise overviews  
  • Step 3: Analyze research trends related to transformer models over the past year
```

### MCP Server Testing

```bash
# Start MCP server in background
python -m egile_researcher.server &

# Test MCP client functionality
python tmp/test_mcp_server.py

# Stop server
pkill -f "egile_researcher.server"
```

## 🔧 Development

### Setting up Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd egile-researcher
poetry install --with dev

# Run tests
poetry run pytest

# Test AI agent capabilities
python tmp/demo_planning.py

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy egile_researcher/
```

### Key Development Files

- `egile_researcher/ai_agent.py` - AI-powered research agent
- `egile_researcher/server.py` - MCP server implementation
- `egile_researcher/agent.py` - Traditional research agent
- `egile_researcher/tools/` - Individual research tools
- `tmp/demo_*.py` - Example and test scripts

## 📖 Documentation

- [MCP Setup Guide](MCP_SETUP_NEW.md) - Comprehensive MCP integration guide
- [API Reference](docs/api.md) - Complete API documentation
- [Configuration Guide](docs/configuration.md) - Configuration options
- [Research Sources](docs/sources.md) - Supported research databases
- [Examples](examples/) - Usage examples and tutorials

## 🚀 What's New

### v0.2.0 - MCP Integration & AI Agents
- **🧠 AI Research Agent**: Intelligent planning and execution of research tasks
- **🌐 MCP Server**: Standardized research tools accessible via MCP protocol
- **🔌 Claude Desktop Integration**: Use research tools directly in Claude Desktop
- **📋 Smart Planning**: LLM-powered planning with fallback heuristics
- **🔄 Dynamic Tool Discovery**: Agents adapt to available research capabilities

### Migration from v0.1.x

The traditional `ResearchAgent` API remains unchanged. New features:

```python
# Old way (still works)
agent = ResearchAgent()
papers = await agent.search_papers("ML")

# New AI-powered way  
result = await ai_research("Find and analyze recent ML papers")

# New MCP integration
# Start server: python -m egile_researcher.server
# Use with Claude Desktop or custom MCP clients
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [MCP Protocol](https://modelcontextprotocol.io/) - Model Context Protocol specification
- [FastMCP](https://github.com/jlowin/fastmcp) - Fast MCP server framework
- [Claude Desktop](https://claude.ai/download) - AI assistant with MCP support

## 🎯 Use Cases

### 📚 Academic Researchers
- Track latest developments in your field
- Automated literature reviews
- Research trend analysis
- Citation impact tracking

### 🏢 R&D Teams
- Competitive intelligence
- Technology trend monitoring
- Patent landscape analysis
- Innovation opportunity identification

### 🤖 AI Applications
- Research tool integration via MCP
- Automated research assistants
- Knowledge base updates
- Content generation pipelines

### 🔬 Research Institutions
- Departmental research dashboards
- Grant application support
- Collaboration opportunity identification
- Research impact assessment

---

**Stay ahead in research with Egile Researcher - where AI meets academic excellence.** 🎓✨

## ✅ Current Status

**The MCP refactoring has been successfully completed!** 🎉

### Tested & Working Features
- ✅ **MCP Server**: Starts correctly and exposes 7 research tools
- ✅ **AI Agent Connection**: Successfully connects to MCP server and discovers tools
- ✅ **Tool Discovery**: All 7 tools properly discovered with descriptions
- ✅ **AI Planning**: LLM-powered planning working with OpenAI integration
- ✅ **Fallback Planning**: Heuristic planning works when LLM unavailable
- ✅ **Multi-Step Workflows**: Intelligent agents can plan complex research tasks

### Architecture Transformation Complete
- ✅ Migrated from monolithic agent to MCP server architecture
- ✅ AI agents with dynamic tool discovery
- ✅ Traditional API maintained for backward compatibility
- ✅ Full documentation and examples updated

### Ready for Production
The system is ready for use in all supported modes: AI agents, traditional agents, MCP server, and client integrations.

---
