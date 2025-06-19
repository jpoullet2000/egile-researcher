"""
Configuration classes for the Egile Researcher package.

These classes provide type-safe configuration management
following Azure best practices for research automation.
"""

import os
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI client."""

    endpoint: str
    api_version: str = "2024-12-01-preview"
    key_vault_url: Optional[str] = None
    api_key_secret_name: Optional[str] = None
    default_model: str = "gpt-4.1-mini"
    max_retries: int = 3
    timeout: int = 30
    use_managed_identity: bool = True

    @classmethod
    def from_environment(cls, env_file: Optional[str] = None) -> "AzureOpenAIConfig":
        """Load configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load from .env file in current directory
            load_dotenv()

        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            key_vault_url=os.getenv("AZURE_KEY_VAULT_URL"),
            api_key_secret_name=os.getenv("AZURE_OPENAI_API_KEY_SECRET_NAME"),
            default_model=os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4"),
            max_retries=int(os.getenv("AZURE_OPENAI_MAX_RETRIES", "3")),
            timeout=int(os.getenv("AZURE_OPENAI_TIMEOUT", "30")),
            use_managed_identity=os.getenv("AZURE_USE_MANAGED_IDENTITY", "true").lower()
            == "true",
        )


class ResearchAgentConfig(BaseModel):
    """Configuration for the Research Agent."""

    name: str = Field(default="ResearchAgent", description="Agent name")
    description: str = Field(
        default="Intelligent research automation agent",
        description="Agent description",
    )
    openai_config: AzureOpenAIConfig = Field(
        ..., description="Azure OpenAI configuration"
    )

    # Research-specific configuration
    research_areas: List[str] = Field(
        default_factory=list, description="Default research areas of interest"
    )
    max_papers_per_search: int = Field(
        default=20, description="Maximum number of papers to fetch per search"
    )
    days_back_default: int = Field(
        default=30, description="Default number of days to look back for papers"
    )

    # AI models for different tasks
    summarization_model: str = Field(
        default="gpt-4.1-mini", description="Model for paper summarization"
    )
    analysis_model: str = Field(
        default="gpt-4.1-mini", description="Model for research analysis"
    )
    trend_analysis_model: str = Field(
        default="gpt-4.1-mini", description="Model for trend analysis"
    )

    # Content processing settings
    summary_max_tokens: int = Field(
        default=1000, description="Maximum tokens for summaries"
    )
    analysis_max_tokens: int = Field(
        default=2000, description="Maximum tokens for analysis"
    )
    summary_temperature: float = Field(
        default=0.3, description="Temperature for summarization"
    )
    analysis_temperature: float = Field(
        default=0.5, description="Temperature for analysis"
    )

    @validator("research_areas")
    def validate_research_areas(cls, v):
        if not isinstance(v, list):
            raise ValueError("Research areas must be a list")
        return v


class PaperSummarizationConfig(BaseModel):
    """Configuration for paper summarization tools."""

    model: str = Field(default="gpt-4", description="Model for summarization")
    max_tokens: int = Field(default=1000, description="Maximum tokens for summary")
    temperature: float = Field(default=0.3, description="Temperature for summarization")

    # Summary structure settings
    include_methodology: bool = Field(
        default=True, description="Include methodology in summary"
    )
    include_results: bool = Field(
        default=True, description="Include results in summary"
    )
    include_limitations: bool = Field(
        default=True, description="Include limitations in summary"
    )
    include_future_work: bool = Field(
        default=False, description="Include future work suggestions"
    )

    # Content extraction settings
    extract_figures: bool = Field(
        default=False, description="Extract and describe figures"
    )
    extract_tables: bool = Field(
        default=False, description="Extract and describe tables"
    )


class ResearchSourceConfig(BaseModel):
    """Configuration for research paper sources."""

    # Source priorities (1-10, higher = preferred)
    arxiv_priority: int = Field(default=9, description="Priority for arXiv papers")
    pubmed_priority: int = Field(default=8, description="Priority for PubMed papers")
    semantic_scholar_priority: int = Field(
        default=7, description="Priority for Semantic Scholar"
    )
    crossref_priority: int = Field(default=6, description="Priority for CrossRef")

    # API configuration
    arxiv_base_url: str = Field(
        default="http://export.arxiv.org/api/query", description="arXiv API endpoint"
    )
    pubmed_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        description="PubMed API endpoint",
    )

    # Rate limiting
    requests_per_second: float = Field(
        default=1.0, description="Maximum requests per second to APIs"
    )
    max_concurrent_requests: int = Field(
        default=5, description="Maximum concurrent requests"
    )


class TrendAnalysisConfig(BaseModel):
    """Configuration for research trend analysis."""

    model: str = Field(default="gpt-4.1-mini", description="Model for trend analysis")
    analysis_window_days: int = Field(
        default=90, description="Time window for trend analysis in days"
    )
    min_papers_for_trend: int = Field(
        default=5, description="Minimum papers needed to identify a trend"
    )

    # Clustering settings
    enable_clustering: bool = Field(default=True, description="Enable topic clustering")
    max_clusters: int = Field(
        default=10, description="Maximum number of topic clusters"
    )

    # Visualization settings
    generate_plots: bool = Field(
        default=True, description="Generate trend visualization plots"
    )
    plot_format: str = Field(default="png", description="Format for generated plots")


class ExportConfig(BaseModel):
    """Configuration for report and data export."""

    default_format: str = Field(default="markdown", description="Default export format")
    supported_formats: List[str] = Field(
        default_factory=lambda: ["markdown", "pdf", "json", "csv"],
        description="Supported export formats",
    )

    # Report settings
    include_metadata: bool = Field(
        default=True, description="Include paper metadata in exports"
    )
    include_summaries: bool = Field(
        default=True, description="Include summaries in exports"
    )
    include_analysis: bool = Field(
        default=True, description="Include analysis in exports"
    )

    # File settings
    output_directory: str = Field(
        default="./research_reports", description="Default output directory"
    )
    filename_template: str = Field(
        default="research_report_{timestamp}_{topic}",
        description="Template for generated filenames",
    )

    @validator("default_format")
    def validate_format(cls, v, values):
        supported = values.get("supported_formats", [])
        if v not in supported:
            raise ValueError(f"Format {v} not in supported formats: {supported}")
        return v
