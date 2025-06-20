# Configuration Guide

Complete configuration guide for Egile Researcher.

## Environment Variables

### Required Azure OpenAI Configuration

```bash
# Azure OpenAI API credentials
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/

# API Configuration
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEFAULT_MODEL=gpt-4.1-mini

# Authentication Method
AZURE_USE_MANAGED_IDENTITY=false  # Set to true for Azure-hosted environments
```

### Optional Azure Key Vault Configuration

For secure credential storage in production:

```bash
# Key Vault Configuration
AZURE_KEY_VAULT_URL=https://your-keyvault.vault.azure.net/
AZURE_OPENAI_API_KEY_SECRET_NAME=openai-api-key

# Service Principal (for CI/CD)
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id
```

### Research Database APIs (Optional)

```bash
# Research Source APIs
ARXIV_API_KEY=your_arxiv_key
PUBMED_API_KEY=your_pubmed_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key
GOOGLE_SCHOLAR_API_KEY=your_google_scholar_key
```

### Agent Configuration

```bash
# Research Agent Settings
RESEARCH_AGENT_NAME=MyResearchAgent
RESEARCH_AGENT_DESCRIPTION="AI research assistant"
RESEARCH_MAX_PAPERS_PER_SEARCH=20
RESEARCH_DAYS_BACK_DEFAULT=30

# AI Model Configuration
SUMMARIZATION_MODEL=gpt-4.1-mini
ANALYSIS_MODEL=gpt-4.1-mini
TREND_ANALYSIS_MODEL=gpt-4.1-mini

# Token Limits
SUMMARY_MAX_TOKENS=1000
ANALYSIS_MAX_TOKENS=2000

# Temperature Settings
SUMMARY_TEMPERATURE=0.3
ANALYSIS_TEMPERATURE=0.5
```

---

## Configuration Files

### `.env` File Setup

Create a `.env` file in your project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=2d7db20c19364d6fb1a8331adf19fe6b
AZURE_OPENAI_ENDPOINT=https://gpttestingsweden.openai.azure.com/
AZURE_USE_MANAGED_IDENTITY=false
AZURE_OPENAI_DEFAULT_MODEL=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Research Configuration
RESEARCH_MAX_PAPERS_PER_SEARCH=15
RESEARCH_DAYS_BACK_DEFAULT=7
```

### Programmatic Configuration

#### Basic Configuration

```python
from egile_researcher.config import AzureOpenAIConfig, ResearchAgentConfig

# Azure OpenAI configuration
openai_config = AzureOpenAIConfig(
    endpoint="https://your-endpoint.openai.azure.com/",
    api_version="2024-12-01-preview",
    default_model="gpt-4.1-mini",
    use_managed_identity=False
)

# Research agent configuration
agent_config = ResearchAgentConfig(
    openai_config=openai_config,
    research_areas=["machine learning", "artificial intelligence"],
    max_papers_per_search=10,
    days_back_default=14
)
```

#### Advanced Configuration

```python
from egile_researcher.config import AzureOpenAIConfig, ResearchAgentConfig

# Advanced Azure OpenAI configuration
openai_config = AzureOpenAIConfig(
    endpoint="https://your-endpoint.openai.azure.com/",
    api_version="2024-12-01-preview",
    key_vault_url="https://your-keyvault.vault.azure.net/",
    api_key_secret_name="openai-api-key",
    default_model="gpt-4.1-mini",
    max_retries=5,
    timeout=60,
    use_managed_identity=True
)

# Comprehensive research agent configuration
agent_config = ResearchAgentConfig(
    name="AdvancedResearchAgent",
    description="Advanced AI research assistant with specialized capabilities",
    openai_config=openai_config,
    research_areas=[
        "machine learning",
        "natural language processing",
        "computer vision",
        "robotics"
    ],
    max_papers_per_search=25,
    days_back_default=30,
    summarization_model="gpt-4.1-mini",
    analysis_model="gpt-4.1-mini",
    trend_analysis_model="gpt-4.1-mini",
    summary_max_tokens=1500,
    analysis_max_tokens=2500,
    summary_temperature=0.2,
    analysis_temperature=0.4
)
```

---

## MCP Server Configuration

### Server Environment

```bash
# MCP Server specific settings
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000
MCP_SERVER_LOG_LEVEL=INFO

# Research tool configuration
ENABLE_ARXIV_SEARCH=true
ENABLE_PUBMED_SEARCH=true
ENABLE_SEMANTIC_SCHOLAR=true
ENABLE_GOOGLE_SCHOLAR=false  # Requires API key

# Rate limiting
MAX_CONCURRENT_SEARCHES=5
RATE_LIMIT_PER_MINUTE=60
```

### Claude Desktop Integration

Add to your Claude Desktop configuration file:

**Location:** `~/.config/claude/claude_desktop_config.json` (Linux/Mac) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

```json
{
  "mcpServers": {
    "egile-researcher": {
      "command": "python",
      "args": ["-m", "egile_researcher.server"],
      "env": {
        "AZURE_OPENAI_API_KEY": "your-api-key",
        "AZURE_OPENAI_ENDPOINT": "https://your-endpoint.openai.azure.com/",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
        "AZURE_USE_MANAGED_IDENTITY": "false"
      }
    }
  }
}
```

### VS Code Integration

For VS Code MCP extension:

```json
{
  "mcp.servers": {
    "egile-researcher": {
      "command": ["python", "-m", "egile_researcher.server"],
      "args": [],
      "env": {
        "AZURE_OPENAI_API_KEY": "your-api-key",
        "AZURE_OPENAI_ENDPOINT": "your-endpoint"
      }
    }
  }
}
```

---

## Authentication Methods

### 1. API Key Authentication (Development)

Simplest method for development and testing:

```python
config = AzureOpenAIConfig(
    endpoint="https://your-endpoint.openai.azure.com/",
    use_managed_identity=False
)
# Set AZURE_OPENAI_API_KEY environment variable
```

### 2. Managed Identity (Azure-hosted)

For applications running on Azure (recommended for production):

```python
config = AzureOpenAIConfig(
    endpoint="https://your-endpoint.openai.azure.com/",
    use_managed_identity=True
)
```

### 3. Service Principal (CI/CD)

For automated deployments and CI/CD pipelines:

```python
config = AzureOpenAIConfig(
    endpoint="https://your-endpoint.openai.azure.com/",
    use_managed_identity=True  # DefaultAzureCredential handles service principal
)
# Set AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
```

### 4. Azure Key Vault (Production)

For secure credential storage:

```python
config = AzureOpenAIConfig(
    endpoint="https://your-endpoint.openai.azure.com/",
    key_vault_url="https://your-keyvault.vault.azure.net/",
    api_key_secret_name="openai-api-key",
    use_managed_identity=True
)
```

---

## Configuration Validation

### Validate Configuration

```python
from egile_researcher.config import AzureOpenAIConfig, ResearchAgentConfig

try:
    # Load configuration
    openai_config = AzureOpenAIConfig.from_environment()
    agent_config = ResearchAgentConfig(openai_config=openai_config)
    
    print("✅ Configuration is valid")
    print(f"Endpoint: {openai_config.endpoint}")
    print(f"Model: {openai_config.default_model}")
    print(f"Max papers per search: {agent_config.max_papers_per_search}")
    
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

### Test Configuration

```python
from egile_researcher import ResearchAgent

async def test_config():
    try:
        agent = ResearchAgent()
        # Test a simple operation
        papers = await agent.search_papers("test", max_results=1)
        print("✅ Configuration test successful")
        await agent.close()
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

import asyncio
asyncio.run(test_config())
```

---

## Performance Tuning

### Model Selection

Choose appropriate models for different tasks:

```python
config = ResearchAgentConfig(
    summarization_model="gpt-4.1-mini",      # Fast, cost-effective
    analysis_model="gpt-4.1-mini",          # Balanced performance
    trend_analysis_model="gpt-4.1-mini"     # High-quality analysis
)
```

### Token Management

Optimize token usage:

```python
config = ResearchAgentConfig(
    summary_max_tokens=800,      # Concise summaries
    analysis_max_tokens=1500,    # Detailed analysis
    summary_temperature=0.1,     # More deterministic
    analysis_temperature=0.3     # Slightly more creative
)
```

### Rate Limiting

Configure rate limits for API calls:

```python
openai_config = AzureOpenAIConfig(
    max_retries=3,
    timeout=30,
    # Use backoff strategies in production
)
```

---

## Troubleshooting

### Common Configuration Issues

1. **Authentication Errors**
   - Check API key validity
   - Verify endpoint URL
   - Ensure managed identity is properly configured

2. **Model Not Found**
   - Verify model deployment name
   - Check model availability in your region
   - Ensure sufficient quota

3. **Connection Timeouts**
   - Increase timeout values
   - Check network connectivity
   - Verify firewall settings

4. **Rate Limiting**
   - Reduce concurrent requests
   - Implement exponential backoff
   - Check API quotas

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use structlog
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG)
)
```

### Configuration Diagnostics

```python
from egile_researcher.config import AzureOpenAIConfig

def diagnose_config():
    config = AzureOpenAIConfig.from_environment()
    
    print("Configuration Diagnostics:")
    print(f"Endpoint: {config.endpoint}")
    print(f"API Version: {config.api_version}")
    print(f"Default Model: {config.default_model}")
    print(f"Use Managed Identity: {config.use_managed_identity}")
    print(f"Max Retries: {config.max_retries}")
    print(f"Timeout: {config.timeout}")
    
    if config.key_vault_url:
        print(f"Key Vault URL: {config.key_vault_url}")
        print(f"API Key Secret Name: {config.api_key_secret_name}")

diagnose_config()
```
