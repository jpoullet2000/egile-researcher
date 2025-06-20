"""
MCP Server for Egile Researcher

This server exposes research agent capabilities as MCP tools.
"""

import asyncio
from typing import Any, Dict, List, Optional

import structlog
from fastmcp import FastMCP

from .agent import ResearchAgent
from .config import ResearchAgentConfig, AzureOpenAIConfig


logger = structlog.get_logger(__name__)

# Global research agent instance
research_agent: Optional[ResearchAgent] = None

# Initialize MCP server
mcp = FastMCP("Egile Researcher")


async def get_research_agent() -> ResearchAgent:
    """Get or create the research agent instance."""
    global research_agent
    if research_agent is None:
        config = ResearchAgentConfig(openai_config=AzureOpenAIConfig.from_environment())
        research_agent = ResearchAgent(config=config)
        logger.info("Research agent initialized")
    return research_agent


@mcp.tool()
async def search_papers(
    query: str,
    days_back: Optional[int] = None,
    max_results: Optional[int] = None,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for research papers across multiple sources.

    Args:
        query: Search query string
        days_back: Number of days to look back (default from config)
        max_results: Maximum number of results (default from config)
        sources: List of sources to search (arXiv, PubMed, etc.)

    Returns:
        List of paper metadata dictionaries
    """
    agent = await get_research_agent()
    return await agent.search_papers(
        query=query,
        days_back=days_back,
        max_results=max_results,
        sources=sources,
    )


@mcp.tool()
async def summarize_paper(
    paper: Dict[str, Any],
    summary_type: str = "comprehensive",
    include_analysis: bool = True,
) -> Dict[str, Any]:
    """
    Generate an intelligent summary of a research paper.

    Args:
        paper: Paper metadata and content
        summary_type: Type of summary (brief, comprehensive, technical)
        include_analysis: Whether to include analytical insights

    Returns:
        Summary with metadata and analysis
    """
    agent = await get_research_agent()
    return await agent.summarize_paper(
        paper=paper,
        summary_type=summary_type,
        include_analysis=include_analysis,
    )


@mcp.tool()
async def analyze_trends(
    topic: str,
    time_period: str = "last_month",
    max_papers: int = 50,
) -> Dict[str, Any]:
    """
    Analyze research trends for a specific topic.

    Args:
        topic: Research topic to analyze
        time_period: Time period for analysis
        max_papers: Maximum number of papers to analyze

    Returns:
        Trend analysis results
    """
    agent = await get_research_agent()
    return await agent.analyze_trends(
        topic=topic,
        time_period=time_period,
        max_papers=max_papers,
    )


@mcp.tool()
async def compare_papers(
    papers: List[Dict[str, Any]],
    comparison_aspects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple research papers.

    Args:
        papers: List of papers to compare
        comparison_aspects: Specific aspects to compare

    Returns:
        Comparative analysis results
    """
    agent = await get_research_agent()
    return await agent.compare_papers(
        papers=papers,
        comparison_aspects=comparison_aspects,
    )


@mcp.tool()
async def generate_research_report(
    topic: str,
    include_trends: bool = True,
    include_summaries: bool = True,
    max_papers: int = 20,
) -> Dict[str, Any]:
    """
    Generate a comprehensive research report.

    Args:
        topic: Research topic
        include_trends: Include trend analysis
        include_summaries: Include paper summaries
        max_papers: Maximum papers to include

    Returns:
        Complete research report
    """
    agent = await get_research_agent()
    return await agent.generate_research_report(
        topic=topic,
        include_trends=include_trends,
        include_summaries=include_summaries,
        max_papers=max_papers,
    )


@mcp.tool()
async def search_and_summarize(
    query: str,
    max_papers: int = 10,
    summary_type: str = "brief",
    days_back: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Search for papers and generate summaries in one step.

    Args:
        query: Search query string
        max_papers: Maximum number of papers to process
        summary_type: Type of summary (brief, comprehensive, technical)
        days_back: Number of days to look back

    Returns:
        Combined search and summary results
    """
    agent = await get_research_agent()

    # Search for papers
    papers = await agent.search_papers(
        query=query,
        max_results=max_papers,
        days_back=days_back,
    )

    if not papers:
        return {
            "query": query,
            "papers_found": 0,
            "summaries": [],
            "message": "No papers found for the given query",
        }

    # Generate summaries
    summaries = []
    for paper in papers:
        try:
            summary = await agent.summarize_paper(
                paper=paper,
                summary_type=summary_type,
                include_analysis=False,
            )
            summaries.append(summary)
        except Exception as e:
            logger.warning(f"Failed to summarize paper {paper.get('id')}: {e}")
            continue

    return {
        "query": query,
        "papers_found": len(papers),
        "papers_processed": len(summaries),
        "summaries": summaries,
    }


@mcp.tool()
async def analyze_topic_comprehensively(
    topic: str,
    max_papers: int = 30,
    time_period: str = "last_month",
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a research topic including trends and comparisons.

    Args:
        topic: Research topic to analyze
        max_papers: Maximum number of papers to analyze
        time_period: Time period for analysis

    Returns:
        Comprehensive topic analysis
    """
    agent = await get_research_agent()

    # Get papers
    papers = await agent.search_papers(
        query=topic,
        max_results=max_papers,
        days_back=agent._time_period_to_days(time_period),
    )

    if not papers:
        return {
            "topic": topic,
            "message": "No papers found for comprehensive analysis",
        }

    # Generate trend analysis
    trends = await agent.analyze_trends(
        topic=topic,
        time_period=time_period,
        max_papers=len(papers),
    )

    # Compare top papers if we have enough
    comparison = None
    if len(papers) >= 3:
        top_papers = papers[:3]  # Compare top 3 papers
        comparison = await agent.compare_papers(
            papers=top_papers,
            comparison_aspects=["methodology", "findings", "significance"],
        )

    # Generate summaries for key papers
    key_summaries = []
    for paper in papers[:5]:  # Summarize top 5 papers
        try:
            summary = await agent.summarize_paper(
                paper=paper,
                summary_type="comprehensive",
                include_analysis=True,
            )
            key_summaries.append(summary)
        except Exception as e:
            logger.warning(f"Failed to summarize key paper {paper.get('id')}: {e}")
            continue

    return {
        "topic": topic,
        "time_period": time_period,
        "papers_analyzed": len(papers),
        "trends": trends,
        "comparison": comparison,
        "key_summaries": key_summaries,
        "analysis_completed_at": trends.get("analyzed_at"),
    }


async def cleanup():
    """Cleanup resources when server shuts down."""
    global research_agent
    if research_agent:
        await research_agent.close()
        research_agent = None
    logger.info("MCP server cleanup completed")


def main():
    """Main entry point for the MCP server."""
    import signal
    import sys

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(cleanup())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
