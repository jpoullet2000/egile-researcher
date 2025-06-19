"""
Research Agent for intelligent research automation and paper analysis.

This agent provides intelligent routing and execution of research tasks
across different data sources and tools, optimized for academic workflows.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

import structlog
import arxiv
from habanero import Crossref

from .client import AzureOpenAIClient
from .config import ResearchAgentConfig, AzureOpenAIConfig
from .exceptions import ResearchAgentError, AzureOpenAIError


logger = structlog.get_logger(__name__)


class ResearchAgent:
    """
    Intelligent research automation agent.

    Features:
    - Intelligent paper discovery and fetching
    - Context-aware summarization
    - Research trend analysis and insights
    - Multi-source integration (arXiv, PubMed, etc.)
    - Comparative analysis across papers
    - Automated report generation
    - Citation network analysis
    """

    def __init__(
        self,
        config: Optional[ResearchAgentConfig] = None,
        openai_config: Optional[AzureOpenAIConfig] = None,
    ):
        """
        Initialize the research agent.

        Args:
            config: Research agent configuration
            openai_config: Azure OpenAI configuration
        """
        self.config = config or ResearchAgentConfig(
            openai_config=openai_config or AzureOpenAIConfig.from_environment()
        )
        self.openai_client = AzureOpenAIClient(self.config.openai_config)

        # Cache for results and optimization
        self._paper_cache: Dict[str, Any] = {}
        self._summary_cache: Dict[str, Any] = {}

        # Initialize external clients
        self._arxiv_client = arxiv.Client()
        self._crossref_client = Crossref()

        logger.info(
            "Initializing Research Agent",
            agent_name=self.config.name,
            research_areas=self.config.research_areas,
            max_papers_per_search=self.config.max_papers_per_search,
        )

    async def search_papers(
        self,
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
        try:
            days_back = days_back or self.config.days_back_default
            max_results = max_results or self.config.max_papers_per_search
            sources = sources or ["arxiv", "pubmed"]

            logger.info(
                "Searching for papers",
                query=query,
                days_back=days_back,
                max_results=max_results,
                sources=sources,
            )

            all_papers = []

            # Search arXiv if included in sources
            if "arxiv" in sources:
                arxiv_papers = await self._search_arxiv(
                    query, max_results // len(sources), days_back
                )
                all_papers.extend(arxiv_papers)

            # Search CrossRef if included in sources
            if "crossref" in sources:
                crossref_papers = await self._search_crossref(
                    query, max_results // len(sources), days_back
                )
                all_papers.extend(crossref_papers)

            # Sort by publication date (newest first)
            all_papers.sort(key=lambda x: x.get("published_date", ""), reverse=True)

            # Limit to max_results
            all_papers = all_papers[:max_results]

            # Cache results
            for paper in all_papers:
                self._paper_cache[paper["id"]] = paper

            logger.info(f"Found {len(all_papers)} papers")
            return all_papers

        except Exception as e:
            logger.error("Paper search failed", error=str(e), query=query)
            raise ResearchAgentError(
                f"Paper search failed: {e}",
                agent_name=self.config.name,
                operation="search_papers",
            )

    async def summarize_paper(
        self,
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
        try:
            paper_id = paper.get("id", "unknown")
            cache_key = f"{paper_id}_{summary_type}_{include_analysis}"

            # Check cache first
            if cache_key in self._summary_cache:
                logger.info("Returning cached summary", paper_id=paper_id)
                return self._summary_cache[cache_key]

            logger.info(
                "Generating paper summary",
                paper_id=paper_id,
                title=paper.get("title", "Unknown"),
                summary_type=summary_type,
            )

            # Extract paper content (in real implementation, would fetch full text)
            paper_content = (
                paper.get("abstract", "") + "\n\n" + paper.get("content", "")
            )
            if not paper_content.strip():
                paper_content = paper.get("abstract", "No content available")

            # Generate summary using OpenAI
            summary = await self.openai_client.summarize_paper(
                paper_content=paper_content,
                paper_metadata=paper,
                summary_type=summary_type,
            )

            # Generate additional analysis if requested
            analysis = None
            if include_analysis:
                analysis = await self._analyze_paper_significance(paper, summary)

            result = {
                "paper_id": paper_id,
                "paper_title": paper.get("title", "Unknown"),
                "summary": summary,
                "summary_type": summary_type,
                "analysis": analysis,
                "key_findings": self._extract_key_findings(summary),
                "methodology": self._extract_methodology(summary),
                "limitations": self._extract_limitations(summary),
                "generated_at": datetime.now().isoformat(),
                "agent_version": "1.0",
            }

            # Cache the result
            self._summary_cache[cache_key] = result

            logger.info("Paper summary generated successfully", paper_id=paper_id)
            return result

        except Exception as e:
            logger.error(
                "Paper summarization failed", error=str(e), paper_id=paper.get("id")
            )
            raise ResearchAgentError(
                f"Paper summarization failed: {e}",
                agent_name=self.config.name,
                operation="summarize_paper",
            )

    async def analyze_trends(
        self,
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
        try:
            logger.info(
                "Analyzing research trends",
                topic=topic,
                time_period=time_period,
                max_papers=max_papers,
            )

            # Search for papers on the topic
            papers = await self.search_papers(
                query=topic,
                max_results=max_papers,
                days_back=self._time_period_to_days(time_period),
            )

            if not papers:
                return {
                    "topic": topic,
                    "time_period": time_period,
                    "message": "No papers found for analysis",
                    "analyzed_at": datetime.now().isoformat(),
                }

            # Generate summaries for trend analysis
            papers_with_summaries = []
            for paper in papers[:20]:  # Limit for API efficiency
                summary_result = await self.summarize_paper(paper, summary_type="brief")
                papers_with_summaries.append(
                    {
                        **paper,
                        "summary": summary_result["summary"],
                    }
                )

            # Analyze trends using OpenAI
            trend_analysis = await self.openai_client.analyze_research_trends(
                papers_data=papers_with_summaries,
                topic_focus=topic,
                time_period=time_period,
            )

            result = {
                "topic": topic,
                "time_period": time_period,
                "papers_analyzed": len(papers_with_summaries),
                "total_papers_found": len(papers),
                "trends": trend_analysis,
                "analyzed_at": datetime.now().isoformat(),
            }

            logger.info("Trend analysis completed", topic=topic)
            return result

        except Exception as e:
            logger.error("Trend analysis failed", error=str(e), topic=topic)
            raise ResearchAgentError(
                f"Trend analysis failed: {e}",
                agent_name=self.config.name,
                operation="analyze_trends",
            )

    async def compare_papers(
        self,
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
        try:
            logger.info(
                "Comparing papers",
                paper_count=len(papers),
                comparison_aspects=comparison_aspects,
            )

            if len(papers) < 2:
                raise ResearchAgentError(
                    "At least 2 papers required for comparison",
                    agent_name=self.config.name,
                    operation="compare_papers",
                )

            # Generate summaries for each paper
            papers_with_summaries = []
            for paper in papers:
                summary_result = await self.summarize_paper(paper, summary_type="brief")
                papers_with_summaries.append(
                    {
                        **paper,
                        "summary": summary_result["summary"],
                    }
                )

            # Perform comparison using OpenAI
            comparison_result = await self.openai_client.compare_papers(
                papers=papers_with_summaries,
                comparison_aspects=comparison_aspects,
            )

            result = {
                "papers_compared": len(papers),
                "paper_titles": [p.get("title", "Unknown") for p in papers],
                "comparison_aspects": comparison_aspects,
                "comparison": comparison_result,
                "compared_at": datetime.now().isoformat(),
            }

            logger.info("Paper comparison completed", paper_count=len(papers))
            return result

        except Exception as e:
            logger.error("Paper comparison failed", error=str(e))
            raise ResearchAgentError(
                f"Paper comparison failed: {e}",
                agent_name=self.config.name,
                operation="compare_papers",
            )

    async def generate_research_report(
        self,
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
        try:
            logger.info(
                "Generating research report",
                topic=topic,
                include_trends=include_trends,
                include_summaries=include_summaries,
                max_papers=max_papers,
            )

            report = {
                "topic": topic,
                "generated_at": datetime.now().isoformat(),
                "agent_name": self.config.name,
                "sections": {},
            }

            # Search for papers
            papers = await self.search_papers(query=topic, max_results=max_papers)
            report["sections"]["papers_found"] = {
                "count": len(papers),
                "papers": papers
                if include_summaries
                else [
                    {
                        k: v
                        for k, v in p.items()
                        if k in ["id", "title", "authors", "published_date"]
                    }
                    for p in papers
                ],
            }

            # Generate summaries if requested
            if include_summaries and papers:
                summaries = []
                for paper in papers[:10]:  # Limit for efficiency
                    summary = await self.summarize_paper(paper)
                    summaries.append(summary)
                report["sections"]["summaries"] = summaries

            # Generate trend analysis if requested
            if include_trends and papers:
                trends = await self.analyze_trends(topic, max_papers=len(papers))
                report["sections"]["trends"] = trends

            logger.info("Research report generated successfully", topic=topic)
            return report

        except Exception as e:
            logger.error("Research report generation failed", error=str(e), topic=topic)
            raise ResearchAgentError(
                f"Research report generation failed: {e}",
                agent_name=self.config.name,
                operation="generate_research_report",
            )

    def _time_period_to_days(self, time_period: str) -> int:
        """Convert time period string to number of days."""
        period_map = {
            "last_week": 7,
            "last_month": 30,
            "last_quarter": 90,
            "last_year": 365,
            "recent": 30,
        }
        return period_map.get(time_period, 30)

    def _extract_key_findings(self, summary: str) -> List[str]:
        """Extract key findings from summary text."""
        # Simple extraction - in real implementation, could use NLP
        findings = []
        lines = summary.split("\n")
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in ["finding", "result", "conclude", "show"]
            ):
                findings.append(line.strip())
        return findings[:3]  # Return top 3

    def _extract_methodology(self, summary: str) -> str:
        """Extract methodology from summary text."""
        lines = summary.split("\n")
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in ["method", "approach", "technique"]
            ):
                return line.strip()
        return "Methodology not clearly identified"

    def _extract_limitations(self, summary: str) -> str:
        """Extract limitations from summary text."""
        lines = summary.split("\n")
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in ["limitation", "constraint", "weakness"]
            ):
                return line.strip()
        return "Limitations not explicitly mentioned"

    async def _analyze_paper_significance(
        self, paper: Dict[str, Any], summary: str
    ) -> Dict[str, Any]:
        """Analyze the significance and impact of a paper."""
        try:
            significance_prompt = f"""Analyze the significance and potential impact of this research paper:

Title: {paper.get("title", "Unknown")}
Summary: {summary}

Provide a brief analysis covering:
1. Research significance
2. Potential impact on the field
3. Novelty of the approach
4. Practical applications

Keep the analysis concise and objective."""

            messages = [
                {"role": "system", "content": "You are a research impact analyst."},
                {"role": "user", "content": significance_prompt},
            ]

            response = await self.openai_client.chat_completion(
                messages=messages, temperature=0.4, max_tokens=500
            )

            return {
                "significance_analysis": response.choices[0].message.content,
                "analyzed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning(f"Significance analysis failed: {e}")
            return {
                "significance_analysis": "Analysis not available",
                "error": str(e),
            }

    async def _search_arxiv(
        self, query: str, max_results: int, days_back: int
    ) -> List[Dict[str, Any]]:
        """Search arXiv for papers."""
        try:
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=days_back)

            # Create arXiv search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            papers = []
            for result in self._arxiv_client.results(search):
                # Filter by date
                if result.published.replace(tzinfo=None) < date_threshold:
                    continue

                paper = {
                    "id": f"arxiv:{result.entry_id.split('/')[-1]}",
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "published_date": result.published.isoformat(),
                    "source": "arxiv",
                    "url": result.entry_id,
                    "keywords": [cat for cat in result.categories],
                    "pdf_url": result.pdf_url,
                }
                papers.append(paper)

            logger.info(f"Found {len(papers)} arXiv papers")
            return papers

        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

    async def _search_crossref(
        self, query: str, max_results: int, days_back: int
    ) -> List[Dict[str, Any]]:
        """Search CrossRef for papers."""
        try:
            # Calculate date threshold
            date_threshold = datetime.now() - timedelta(days=days_back)
            date_filter = date_threshold.strftime("%Y-%m-%d")

            # Search CrossRef
            works = self._crossref_client.works(
                query=query,
                filter={"from-pub-date": date_filter},
                limit=max_results,
                sort="published",
                order="desc",
            )

            papers = []
            for item in works["message"]["items"]:
                try:
                    # Extract publication date
                    pub_date_parts = item.get(
                        "published-print", item.get("published-online", {})
                    ).get("date-parts", [[]])
                    if pub_date_parts and pub_date_parts[0]:
                        pub_date = datetime(
                            year=pub_date_parts[0][0],
                            month=pub_date_parts[0][1]
                            if len(pub_date_parts[0]) > 1
                            else 1,
                            day=pub_date_parts[0][2]
                            if len(pub_date_parts[0]) > 2
                            else 1,
                        )
                    else:
                        continue  # Skip if no publication date

                    # Filter by date
                    if pub_date < date_threshold:
                        continue

                    authors = []
                    for author in item.get("author", []):
                        name_parts = []
                        if "given" in author:
                            name_parts.append(author["given"])
                        if "family" in author:
                            name_parts.append(author["family"])
                        if name_parts:
                            authors.append(" ".join(name_parts))

                    paper = {
                        "id": f"crossref:{item.get('DOI', 'unknown')}",
                        "title": item.get("title", ["Unknown"])[0]
                        if item.get("title")
                        else "Unknown",
                        "authors": authors,
                        "abstract": item.get("abstract", "Abstract not available"),
                        "published_date": pub_date.isoformat(),
                        "source": "crossref",
                        "url": item.get("URL", ""),
                        "doi": item.get("DOI"),
                        "journal": item.get("container-title", [""])[0]
                        if item.get("container-title")
                        else "",
                    }
                    papers.append(paper)

                except Exception as e:
                    logger.warning(f"Error processing CrossRef item: {e}")
                    continue

            logger.info(f"Found {len(papers)} CrossRef papers")
            return papers

        except Exception as e:
            logger.error(f"CrossRef search failed: {e}")
            return []

    async def close(self):
        """Clean up resources."""
        await self.openai_client.close()
        logger.info("Research Agent closed")
