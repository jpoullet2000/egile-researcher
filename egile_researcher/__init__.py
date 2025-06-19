"""
Egile Researcher: AI-powered research automation for summarizing and analyzing recent publications.

This package provides intelligent tools for researchers to discover, summarize, and analyze
academic papers and research publications from various sources including arXiv, PubMed,
Google Scholar, and more.
"""

from .agent import ResearchAgent
from .client import AzureOpenAIClient
from .config import ResearchAgentConfig, AzureOpenAIConfig

__version__ = "0.1.0"
__author__ = "Jean-Baptiste Poullet"
__email__ = "jeanbaptistepoullet@gmail.com"

__all__ = [
    "ResearchAgent",
    "AzureOpenAIClient",
    "ResearchAgentConfig",
    "AzureOpenAIConfig",
]
