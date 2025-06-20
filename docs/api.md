# API Reference

Complete API documentation for Egile Researcher.

## AI Research Agent

### `ai_research(task: str) -> Dict[str, Any]`

Convenience function for performing AI-powered research with automatic planning.

**Parameters:**
- `task` (str): Description of the research task to accomplish

**Returns:**
- `Dict[str, Any]`: Complete research results with plan and execution details

**Example:**
```python
from egile_researcher import ai_research

result = await ai_research("Find recent papers about quantum computing")
print(f"Plan: {len(result['plan'])} steps")
print(f"Success: {result['summary']['successful_steps']}/{result['summary']['total_steps']}")
```

---

### `AIResearchAgent`

Intelligent research agent that uses LLM reasoning to plan and execute MCP tool usage.

#### Constructor

```python
AIResearchAgent(
    config: Optional[ResearchAgentConfig] = None,
    server_command: Optional[str] = None
)
```

**Parameters:**
- `config`: Research agent configuration (defaults to environment-based config)
- `server_command`: Command to start the MCP server (default: "python -m egile_researcher.server")

#### Methods

##### `async connect()`
Connect to the MCP server and discover available tools.

##### `async call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any`
Call a specific MCP tool.

**Parameters:**
- `tool_name`: Name of the tool to call
- `arguments`: Arguments to pass to the tool

**Returns:**
- Tool execution result

##### `async research(task: str) -> Dict[str, Any]`
Perform an intelligent research task with AI planning.

**Parameters:**
- `task`: Description of the research task

**Returns:**
- Complete research results with plan and execution details

**Example:**
```python
from egile_researcher import AIResearchAgent

async with AIResearchAgent() as agent:
    result = await agent.research("Research ML trends")
    papers = await agent.call_tool("search_papers", {"query": "AI", "max_results": 5})
```

---

## Traditional Research Agent

### `ResearchAgent`

Traditional research agent providing direct API access to research capabilities.

#### Constructor

```python
ResearchAgent(
    config: Optional[ResearchAgentConfig] = None,
    openai_config: Optional[AzureOpenAIConfig] = None
)
```

#### Methods

##### `async search_papers(query: str, days_back: Optional[int] = None, max_results: Optional[int] = None, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]`

Search for research papers across multiple sources.

**Parameters:**
- `query`: Search query string
- `days_back`: Number of days to look back (default from config)
- `max_results`: Maximum number of results (default from config)
- `sources`: List of sources to search (arXiv, PubMed, etc.)

**Returns:**
- List of paper metadata dictionaries

##### `async summarize_paper(paper: Dict[str, Any], summary_type: str = "comprehensive", include_analysis: bool = True) -> Dict[str, Any]`

Generate an intelligent summary of a research paper.

**Parameters:**
- `paper`: Paper metadata and content
- `summary_type`: Type of summary ("brief", "comprehensive", "technical")
- `include_analysis`: Whether to include analytical insights

**Returns:**
- Summary with metadata and analysis

##### `async analyze_trends(topic: str, time_period: str = "last_month", max_papers: int = 50) -> Dict[str, Any]`

Analyze research trends for a specific topic.

**Parameters:**
- `topic`: Research topic to analyze
- `time_period`: Time period for analysis
- `max_papers`: Maximum number of papers to analyze

**Returns:**
- Trend analysis results

##### `async compare_papers(papers: List[Dict[str, Any]], comparison_aspects: Optional[List[str]] = None) -> Dict[str, Any]`

Compare multiple research papers.

**Parameters:**
- `papers`: List of papers to compare
- `comparison_aspects`: Specific aspects to compare

**Returns:**
- Comparative analysis results

##### `async close()`

Close the research agent and clean up resources.

**Example:**
```python
from egile_researcher import ResearchAgent

agent = ResearchAgent()
papers = await agent.search_papers("machine learning", max_results=10)
summary = await agent.summarize_paper(papers[0])
await agent.close()
```

---

## Configuration Classes

### `AzureOpenAIConfig`

Configuration for Azure OpenAI client.

#### Attributes

- `endpoint` (str): Azure OpenAI endpoint URL
- `api_version` (str): API version (default: "2024-12-01-preview")
- `key_vault_url` (Optional[str]): Azure Key Vault URL for secure credential storage
- `api_key_secret_name` (Optional[str]): Secret name in Key Vault
- `default_model` (str): Default model name (default: "gpt-4.1-mini")
- `max_retries` (int): Maximum retry attempts (default: 3)
- `timeout` (int): Request timeout in seconds (default: 30)
- `use_managed_identity` (bool): Use managed identity authentication (default: True)

#### Class Methods

##### `from_environment(env_file: Optional[str] = None) -> AzureOpenAIConfig`

Load configuration from environment variables.

**Parameters:**
- `env_file`: Optional path to .env file

**Returns:**
- Configured AzureOpenAIConfig instance

**Environment Variables:**
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_API_VERSION`: API version
- `AZURE_KEY_VAULT_URL`: Key Vault URL
- `AZURE_OPENAI_API_KEY_SECRET_NAME`: API key secret name
- `AZURE_OPENAI_DEFAULT_MODEL`: Default model
- `AZURE_OPENAI_MAX_RETRIES`: Max retries
- `AZURE_OPENAI_TIMEOUT`: Timeout
- `AZURE_USE_MANAGED_IDENTITY`: Use managed identity

---

### `ResearchAgentConfig`

Configuration for the Research Agent.

#### Attributes

- `name` (str): Agent name (default: "ResearchAgent")
- `description` (str): Agent description
- `openai_config` (AzureOpenAIConfig): Azure OpenAI configuration
- `research_areas` (List[str]): Default research areas of interest
- `max_papers_per_search` (int): Maximum papers per search (default: 20)
- `days_back_default` (int): Default days to look back (default: 30)
- `summarization_model` (str): Model for summarization (default: "gpt-4.1-mini")
- `analysis_model` (str): Model for analysis (default: "gpt-4.1-mini")
- `trend_analysis_model` (str): Model for trend analysis (default: "gpt-4.1-mini")
- `summary_max_tokens` (int): Max tokens for summaries (default: 1000)
- `analysis_max_tokens` (int): Max tokens for analysis (default: 2000)
- `summary_temperature` (float): Temperature for summarization (default: 0.3)
- `analysis_temperature` (float): Temperature for analysis (default: 0.5)

---

## MCP Tools

When using the MCP server, the following tools are available:

### `search_papers`

Search for research papers across multiple sources.

**Arguments:**
- `query` (str): Search query string
- `days_back` (Optional[int]): Number of days to look back
- `max_results` (Optional[int]): Maximum number of results
- `sources` (Optional[List[str]]): List of sources to search

### `summarize_paper`

Generate an intelligent summary of a research paper.

**Arguments:**
- `paper` (Dict[str, Any]): Paper metadata and content
- `summary_type` (str): Type of summary (default: "comprehensive")
- `include_analysis` (bool): Whether to include analytical insights (default: True)

### `analyze_trends`

Analyze research trends for a specific topic.

**Arguments:**
- `topic` (str): Research topic to analyze
- `time_period` (str): Time period for analysis (default: "last_month")
- `max_papers` (int): Maximum number of papers to analyze (default: 50)

### `generate_insights`

Generate insights from research data.

**Arguments:**
- `papers` (List[Dict[str, Any]]): List of papers to analyze
- `focus_areas` (Optional[List[str]]): Specific areas to focus on

### `create_bibliography`

Create formatted bibliographies.

**Arguments:**
- `papers` (List[Dict[str, Any]]): List of papers
- `format` (str): Bibliography format (default: "APA")

---

## Exception Classes

### `ResearchAgentError`

Base exception for research agent errors.

### `AzureOpenAIError`

Exception for Azure OpenAI related errors.

### `AuthenticationError`

Exception for authentication failures.

### `RetryableError`

Exception for errors that can be retried.

---

## Return Types

### Paper Metadata

```python
{
    "title": str,
    "authors": List[str],
    "abstract": str,
    "url": str,
    "published_date": str,
    "source": str,
    "doi": Optional[str],
    "arxiv_id": Optional[str],
    "citations": Optional[int]
}
```

### Summary Result

```python
{
    "summary": str,
    "key_findings": List[str],
    "methodology": str,
    "implications": str,
    "limitations": List[str],
    "future_work": List[str],
    "significance_score": float,
    "summary_type": str,
    "analyzed_at": str
}
```

### Trend Analysis Result

```python
{
    "topic": str,
    "time_period": str,
    "total_papers": int,
    "trends": List[Dict[str, Any]],
    "emerging_topics": List[str],
    "key_researchers": List[str],
    "research_gaps": List[str],
    "analysis_summary": str,
    "analyzed_at": str
}
```

### AI Research Result

```python
{
    "task": str,
    "plan": List[Dict[str, Any]],
    "execution_results": List[Dict[str, Any]],
    "summary": {
        "total_steps": int,
        "successful_steps": int,
        "failed_steps": int,
        "tools_used": List[str],
        "has_errors": bool
    }
}
```
