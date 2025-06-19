#!/usr/bin/env python3
"""
Test script for egile-researcher package.

This script demonstrates the main functionality of the research agent:
- Searching for papers from arXiv and CrossRef
- Summarizing papers with AI analysis
- Analyzing research trends
- Comparing multiple papers
"""

import asyncio
import json
from datetime import datetime
import sys

# Add the package to Python path for testing
sys.path.insert(0, "/home/jbp/projects/egile-researcher")

from egile_researcher import ResearchAgent
from egile_researcher.config import ResearchAgentConfig, AzureOpenAIConfig


async def test_research_agent():
    """Test the research agent functionality."""
    print("🔬 Testing Egile Research Agent")
    print("=" * 50)

    try:
        # Configure Azure OpenAI
        print("📝 Configuring Azure OpenAI...")
        openai_config = AzureOpenAIConfig.from_environment()

        # Configure research agent
        agent_config = ResearchAgentConfig(
            openai_config=openai_config,
            name="TestResearchAgent",
            research_areas=["artificial intelligence", "machine learning"],
            max_papers_per_search=5,
            days_back_default=30,
        )

        # Initialize research agent
        print("🤖 Initializing research agent...")
        agent = ResearchAgent(agent_config)

        # Test 1: Search for papers
        print("\n📚 Test 1: Searching for papers on 'transformer models'...")
        papers = await agent.search_papers(
            query="transformer models",
            days_back=60,
            max_results=3,
            sources=["arxiv"],  # Start with just arxiv for testing
        )

        print(f"Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"  {i}. {paper['title'][:80]}...")
            print(
                f"     Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}"
            )
            print(f"     Source: {paper['source']}")
            print(f"     Date: {paper['published_date'][:10]}")
            print()

        if papers:
            # Test 2: Summarize a paper
            print("📄 Test 2: Summarizing the first paper...")
            first_paper = papers[0]
            summary_result = await agent.summarize_paper(
                paper=first_paper, summary_type="comprehensive", include_analysis=True
            )

            print(f"Paper: {summary_result['paper_title']}")
            print(f"Summary type: {summary_result['summary_type']}")
            print(f"Summary: {summary_result['summary'][:200]}...")
            print(f"Key findings: {summary_result['key_findings']}")

            if summary_result["analysis"]:
                print(
                    f"Significance analysis: {summary_result['analysis']['significance_analysis'][:150]}..."
                )

        # Test 3: Trend analysis
        print("\n📈 Test 3: Analyzing research trends...")
        trends = await agent.analyze_trends(
            topic="large language models", time_period="last_month", max_papers=5
        )

        print(f"Topic: {trends['topic']}")
        print(f"Papers analyzed: {trends['papers_analyzed']}")

        # Handle trends data properly (could be string or dict)
        trends_data = trends["trends"]
        if isinstance(trends_data, str):
            print(f"Trends: {trends_data[:200]}...")
        else:
            print(f"Trends: {str(trends_data)[:200]}...")

        # Test 4: Save results to file
        print("\n💾 Test 4: Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_file = (
            f"/home/jbp/projects/egile-researcher/test_results_{timestamp}.json"
        )

        results = {
            "test_run": timestamp,
            "agent_config": {
                "name": agent_config.name,
                "research_areas": agent_config.research_areas,
                "max_papers": agent_config.max_papers_per_search,
            },
            "papers_found": len(papers),
            "first_paper": papers[0] if papers else None,
            "summary_result": summary_result if papers else None,
            "trends_analysis": trends,
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Results saved to: {results_file}")

        # Clean up
        await agent.close()
        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_research_agent())
