# Examples

This directory contains practical examples demonstrating how to use Egile Researcher.

## Available Examples

### Basic Usage
- [`basic_research.py`](basic_research.py) - Simple research agent usage
- [`ai_research_simple.py`](ai_research_simple.py) - AI-powered research with automatic planning

### Advanced Features
- [`multi_source_search.py`](multi_source_search.py) - Search across multiple research databases
- [`trend_analysis.py`](trend_analysis.py) - Analyze research trends and patterns
- [`paper_comparison.py`](paper_comparison.py) - Compare multiple research papers

### MCP Integration
- [`mcp_server_setup.py`](mcp_server_setup.py) - Set up and test MCP server
- [`claude_integration.py`](claude_integration.py) - Integration with Claude Desktop
- [`custom_mcp_client.py`](custom_mcp_client.py) - Custom MCP client implementation

### Configuration
- [`config_examples.py`](config_examples.py) - Various configuration scenarios
- [`authentication_examples.py`](authentication_examples.py) - Different authentication methods

### Real-World Use Cases
- [`literature_review.py`](literature_review.py) - Automated literature review
- [`research_dashboard.py`](research_dashboard.py) - Research monitoring dashboard
- [`citation_tracking.py`](citation_tracking.py) - Track citations and research impact

## Running Examples

1. **Set up environment:**
   ```bash
   cd egile-researcher
   poetry install
   poetry shell
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure OpenAI credentials
   ```

3. **Run an example:**
   ```bash
   python examples/basic_research.py
   ```

## Requirements

All examples require:
- Azure OpenAI API access
- Environment variables configured
- Internet connection for research database access

Optional:
- Research database API keys for enhanced functionality
- MCP server for MCP-related examples
