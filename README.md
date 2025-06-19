# Egile Researcher

🔬 **AI-Powered Research Automation for Academic Publications**

Egile Researcher is an intelligent research assistant that helps researchers, academics, and knowledge workers stay up-to-date with the latest publications in their field. It provides automated summarization, analysis, and insights from recent academic papers and research publications.

## 🚀 Features

### Core Capabilities
- **Publication Discovery**: Automatically find recent papers from arXiv, PubMed, Google Scholar, and other sources
- **Intelligent Summarization**: AI-powered summaries that extract key findings, methodologies, and implications
- **Research Analysis**: Deep analysis of research trends, citation patterns, and emerging topics
- **Comparative Studies**: Compare multiple papers and identify relationships between research works
- **Citation Tracking**: Monitor citations and impact of specific papers or research areas
- **Topic Clustering**: Group related publications and identify research themes

### Tools & Services
- **Paper Fetcher**: Retrieve papers from multiple academic databases
- **Content Summarizer**: Generate concise, accurate summaries of research papers
- **Trend Analyzer**: Identify emerging research trends and hot topics
- **Citation Mapper**: Visualize citation networks and research impact
- **Research Dashboard**: Interactive dashboard for research insights
- **Export Tools**: Generate reports in multiple formats (PDF, Markdown, JSON)

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

### Basic Usage

```python
import asyncio
from egile_researcher import ResearchAgent
from egile_researcher.config import ResearchAgentConfig, AzureOpenAIConfig

async def main():
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

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Analysis

```python
# Analyze research trends
trends = await agent.analyze_trends(
    topic="transformer architectures",
    time_period="last_month"
)

# Generate comparative analysis
comparison = await agent.compare_papers([paper1, paper2, paper3])

# Create research report
report = await agent.generate_research_report(
    topic="recent advances in NLP",
    include_visualizations=True
)
```

## 🏗️ Architecture

### Core Components

- **Research Agent**: Main orchestrator for research tasks
- **Publication Fetchers**: Interfaces to various academic databases
- **Content Processors**: Extract and process paper content
- **AI Summarizers**: Generate intelligent summaries and analysis
- **Trend Analyzers**: Identify patterns and trends in research
- **Export Modules**: Generate reports and visualizations

### MCP Server Integration

Egile Researcher includes an MCP (Model Context Protocol) server that exposes research capabilities as tools for AI assistants and other applications.

## 🔧 Development

### Setting up Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd egile-researcher
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy egile_researcher/
```

## 📖 Documentation

- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Research Sources](docs/sources.md)
- [Examples](examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [egile-marketing](../egile-marketing): AI-powered marketing automation
- [MCP Protocol](https://modelcontextprotocol.io/): Model Context Protocol specification

---

**Stay ahead in research with Egile Researcher - where AI meets academic excellence.** 🎓
