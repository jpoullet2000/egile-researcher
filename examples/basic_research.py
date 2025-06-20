"""
Basic Research Agent Example

This example demonstrates basic usage of the traditional ResearchAgent
for searching, summarizing, and analyzing research papers.
"""

import asyncio
from egile_researcher import ResearchAgent
from egile_researcher.config import ResearchAgentConfig, AzureOpenAIConfig


async def main():
    print("🔬 Egile Researcher - Basic Usage Example")
    print("=" * 50)

    try:
        # Initialize the research agent
        print("Initializing research agent...")
        openai_config = AzureOpenAIConfig.from_environment()
        agent_config = ResearchAgentConfig(
            openai_config=openai_config,
            research_areas=["machine learning", "artificial intelligence"],
            max_papers_per_search=5,
            days_back_default=30,
        )

        agent = ResearchAgent(agent_config)
        print("✅ Research agent initialized")

        # Search for papers
        print("\n📚 Searching for papers...")
        papers = await agent.search_papers(
            query="large language models", days_back=7, max_results=3
        )

        print(f"✅ Found {len(papers)} papers")

        # Summarize each paper
        print("\n📝 Generating summaries...")
        for i, paper in enumerate(papers, 1):
            print(f"\n--- Paper {i} ---")
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper.get('authors', ['Unknown']))}")
            print(f"Published: {paper.get('published_date', 'Unknown')}")
            print(f"Source: {paper.get('source', 'Unknown')}")

            # Generate summary
            summary = await agent.summarize_paper(
                paper, summary_type="brief", include_analysis=True
            )

            print(f"\nSummary: {summary['summary']}")
            if summary.get("key_findings"):
                print(f"Key Findings: {', '.join(summary['key_findings'])}")

            print("-" * 60)

        # Analyze trends
        print("\n📈 Analyzing research trends...")
        trends = await agent.analyze_trends(
            topic="large language models", time_period="last_month", max_papers=20
        )

        print(f"✅ Trend analysis completed")
        print(f"Total papers analyzed: {trends.get('total_papers', 0)}")
        print(f"Emerging topics: {', '.join(trends.get('emerging_topics', [])[:3])}")

        # Compare papers (if we have multiple)
        if len(papers) >= 2:
            print("\n🔄 Comparing papers...")
            comparison = await agent.compare_papers(papers[:2])
            print(f"✅ Comparison completed")
            print(
                f"Comparison summary: {comparison.get('summary', 'Not available')[:200]}..."
            )

        print("\n🎉 Example completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        if "agent" in locals():
            await agent.close()
            print("🔒 Research agent closed")


if __name__ == "__main__":
    asyncio.run(main())
