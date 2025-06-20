# Research Sources

Comprehensive guide to research databases and sources supported by Egile Researcher.

## Supported Research Databases

### 📚 arXiv

**Description:** Open-access repository of electronic preprints in physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering, systems science, and economics.

**Coverage:**
- Computer Science
- Mathematics
- Physics
- Quantitative Biology
- Quantitative Finance
- Statistics
- Electrical Engineering

**Features:**
- Real-time access to latest preprints
- Full-text search capabilities
- Author and title searching
- Category-based filtering
- No API key required

**Search Examples:**
```python
# Search by category
papers = await agent.search_papers(
    query="machine learning",
    sources=["arxiv"],
    max_results=20
)

# Search specific arXiv categories
papers = await agent.search_papers(
    query="cs.AI OR cs.LG",  # AI or Machine Learning
    sources=["arxiv"]
)
```

---

### 🧬 PubMed

**Description:** Biomedical literature database maintained by the National Library of Medicine, covering life sciences and biomedical topics.

**Coverage:**
- Medicine
- Biomedical Sciences
- Life Sciences
- Public Health
- Clinical Research

**Features:**
- MeSH (Medical Subject Headings) indexing
- Abstract and citation information
- Author and affiliation data
- Publication date filtering
- API key recommended for higher rate limits

**Configuration:**
```bash
PUBMED_API_KEY=your_pubmed_key  # Optional but recommended
```

**Search Examples:**
```python
# Biomedical research
papers = await agent.search_papers(
    query="COVID-19 machine learning",
    sources=["pubmed"],
    days_back=30
)

# Medical imaging
papers = await agent.search_papers(
    query="medical imaging deep learning",
    sources=["pubmed"]
)
```

---

### 🎓 Semantic Scholar

**Description:** AI-powered academic search engine providing access to scientific literature across multiple disciplines.

**Coverage:**
- Computer Science
- Neuroscience
- Biomedical Sciences
- All academic disciplines

**Features:**
- AI-powered paper recommendations
- Citation analysis
- Author networks
- Influential citations
- Paper abstracts and full metadata

**Configuration:**
```bash
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key  # Recommended
```

**Search Examples:**
```python
# AI research with citation analysis
papers = await agent.search_papers(
    query="transformer models NLP",
    sources=["semantic_scholar"],
    max_results=15
)

# Neuroscience research
papers = await agent.search_papers(
    query="brain computer interface",
    sources=["semantic_scholar"]
)
```

---

### 🔍 Google Scholar (Limited)

**Description:** Web search engine that indexes scholarly literature across various disciplines and sources.

**Coverage:**
- All academic disciplines
- Conference papers
- Theses
- Books
- Technical reports

**Features:**
- Broad coverage across disciplines
- Citation counts
- Related articles
- Author profiles

**Limitations:**
- Rate limiting without API key
- No official public API
- Use sparingly to avoid blocking

**Configuration:**
```bash
GOOGLE_SCHOLAR_API_KEY=your_google_scholar_key  # Required for production use
ENABLE_GOOGLE_SCHOLAR=false  # Disabled by default
```

**Search Examples:**
```python
# Enable only with proper API access
papers = await agent.search_papers(
    query="quantum computing algorithms",
    sources=["google_scholar"],
    max_results=10
)
```

---

## Source Selection Strategies

### Multi-Source Search

Search across multiple sources for comprehensive coverage:

```python
# Comprehensive search across multiple sources
papers = await agent.search_papers(
    query="artificial intelligence ethics",
    sources=["arxiv", "pubmed", "semantic_scholar"],
    max_results=30
)
```

### Domain-Specific Search

Choose sources based on research domain:

```python
# Computer Science research
cs_papers = await agent.search_papers(
    query="machine learning interpretability",
    sources=["arxiv", "semantic_scholar"]
)

# Biomedical research
bio_papers = await agent.search_papers(
    query="CRISPR gene editing",
    sources=["pubmed", "semantic_scholar"]
)

# Physics research
physics_papers = await agent.search_papers(
    query="quantum entanglement",
    sources=["arxiv"]
)
```

---

## Search Query Optimization

### arXiv Search Tips

```python
# Category-specific search
papers = await agent.search_papers(
    query="cat:cs.AI AND machine learning",
    sources=["arxiv"]
)

# Author search
papers = await agent.search_papers(
    query="au:Bengio AND deep learning",
    sources=["arxiv"]
)

# Date range search
papers = await agent.search_papers(
    query="transformer architecture",
    sources=["arxiv"],
    days_back=90  # Last 3 months
)
```

### PubMed Search Tips

```python
# MeSH terms
papers = await agent.search_papers(
    query="Artificial Intelligence[MeSH] AND COVID-19",
    sources=["pubmed"]
)

# Field-specific search
papers = await agent.search_papers(
    query="machine learning[Title/Abstract]",
    sources=["pubmed"]
)

# Publication type filter
papers = await agent.search_papers(
    query="deep learning AND Review[ptyp]",
    sources=["pubmed"]
)
```

### Semantic Scholar Search Tips

```python
# Venue-specific search
papers = await agent.search_papers(
    query="venue:NeurIPS machine learning",
    sources=["semantic_scholar"]
)

# Highly cited papers
papers = await agent.search_papers(
    query="neural networks citationCount:>100",
    sources=["semantic_scholar"]
)
```

---

## Rate Limits and Best Practices

### API Rate Limits

| Source | Rate Limit | API Key Required | Notes |
|--------|------------|------------------|-------|
| arXiv | 1 request/3 seconds | No | Built-in throttling |
| PubMed | 3 requests/second | No (10/sec with key) | Higher limits with API key |
| Semantic Scholar | 100 requests/5 minutes | No (higher with key) | Recommended for heavy usage |
| Google Scholar | Very limited | Yes | Use sparingly |

### Best Practices

1. **Use API Keys**: Always configure API keys for better rate limits
2. **Implement Caching**: Cache search results to avoid repeated queries
3. **Batch Requests**: Group related searches when possible
4. **Respect Rate Limits**: Implement proper backoff strategies
5. **Monitor Usage**: Track API usage to avoid quota exhaustion

### Rate Limiting Configuration

```python
from egile_researcher.config import ResearchAgentConfig

config = ResearchAgentConfig(
    max_papers_per_search=20,  # Reasonable default
    days_back_default=30,      # Recent papers
    # Configure retry behavior
    openai_config=AzureOpenAIConfig(
        max_retries=3,
        timeout=30
    )
)
```

---

## Data Quality and Filtering

### Source Quality Indicators

Each source provides different quality indicators:

**arXiv:**
- Submission date
- Version history
- Subject categories
- Author affiliations

**PubMed:**
- Peer review status
- Journal impact factor
- MeSH indexing quality
- Publication type

**Semantic Scholar:**
- Citation count
- Influential citations
- Author h-index
- Venue ranking

### Filtering Strategies

```python
# Filter by publication date
recent_papers = await agent.search_papers(
    query="machine learning",
    days_back=30,  # Last 30 days only
    sources=["arxiv", "semantic_scholar"]
)

# Filter by result quality
high_quality_papers = await agent.search_papers(
    query="deep learning",
    max_results=10,  # Top 10 most relevant
    sources=["semantic_scholar"]  # Use AI-powered ranking
)
```

---

## Custom Source Integration

### Adding New Sources

To integrate additional research databases:

1. **Create Source Interface**
```python
from egile_researcher.tools.base import BasePaperSource

class CustomSource(BasePaperSource):
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    async def search_papers(self, query: str, **kwargs):
        # Implement search logic
        pass
```

2. **Register Source**
```python
from egile_researcher.tools import register_source

register_source("custom_source", CustomSource)
```

3. **Use Custom Source**
```python
papers = await agent.search_papers(
    query="research topic",
    sources=["custom_source"]
)
```

---

## Troubleshooting Source Issues

### Common Issues

1. **API Key Errors**
   ```python
   # Check API key configuration
   import os
   print(f"PubMed API Key: {os.getenv('PUBMED_API_KEY', 'Not set')}")
   ```

2. **Rate Limiting**
   ```python
   # Implement exponential backoff
   import asyncio
   
   async def search_with_backoff(query, max_retries=3):
       for attempt in range(max_retries):
           try:
               return await agent.search_papers(query)
           except RateLimitError:
               await asyncio.sleep(2 ** attempt)
       raise
   ```

3. **Connection Issues**
   ```python
   # Test source connectivity
   async def test_sources():
       for source in ["arxiv", "pubmed", "semantic_scholar"]:
           try:
               papers = await agent.search_papers(
                   query="test", 
                   sources=[source], 
                   max_results=1
               )
               print(f"✅ {source}: Connected")
           except Exception as e:
               print(f"❌ {source}: {e}")
   ```

### Debug Mode

Enable debug logging for source-specific issues:

```python
import logging
logging.getLogger("egile_researcher.tools").setLevel(logging.DEBUG)
```

---

## Future Source Integrations

### Planned Integrations

- **IEEE Xplore**: Engineering and technology papers
- **ACM Digital Library**: Computer science publications
- **SpringerLink**: Scientific publications across disciplines
- **Nature**: High-impact scientific journals
- **JSTOR**: Academic articles and books

### Source Priority Roadmap

1. **High Priority**: IEEE Xplore, ACM Digital Library
2. **Medium Priority**: SpringerLink, Nature
3. **Future Consideration**: JSTOR, discipline-specific databases

To request new source integrations, please create an issue in the project repository with:
- Source name and description
- API documentation links
- Use case and importance
- Sample search queries
