"""
AI Research Simple Example

This example demonstrates the AI-powered research agent that automatically
plans and executes research tasks using LLM reasoning.
"""

import asyncio
from egile_researcher import ai_research, AIResearchAgent


async def simple_ai_research():
    """Simple AI research using the convenience function."""
    print("🤖 AI Research - Simple Example")
    print("=" * 40)

    try:
        # Simple AI research with automatic planning
        result = await ai_research("Find recent papers about quantum machine learning")

        # Display the plan
        print(f"🎯 AI created plan with {len(result['plan'])} steps:")
        for step in result["plan"]:
            print(f"  Step {step['step']}: {step['description']}")

        # Display execution summary
        summary = result["summary"]
        print(f"\n📊 Execution Summary:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Successful: {summary['successful_steps']}")
        print(f"  Failed: {summary['failed_steps']}")
        print(f"  Tools used: {', '.join(summary['tools_used'])}")

        if summary["has_errors"]:
            print("⚠️  Some steps failed - check detailed results")
        else:
            print("✅ All steps completed successfully!")

        print("\n🎉 Simple AI research completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


async def advanced_ai_research():
    """Advanced AI research with manual control."""
    print("\n🧠 AI Research - Advanced Example")
    print("=" * 40)

    try:
        async with AIResearchAgent() as agent:
            print("✅ Connected to MCP server")
            print(f"🛠️  Available tools: {', '.join(agent.available_tools.keys())}")

            # Manual tool calling
            print("\n🔍 Manual tool usage:")
            papers = await agent.call_tool(
                "search_papers",
                {
                    "query": "transformer models in natural language processing",
                    "max_results": 3,
                },
            )
            print(f"Found {len(papers) if isinstance(papers, list) else 1} papers")

            # AI-planned research
            print("\n🎯 AI-planned research:")
            result = await agent.research(
                "Analyze trends in computer vision research over the past year"
            )

            print(f"Plan had {len(result['plan'])} steps:")
            for step in result["plan"]:
                print(f"  • {step['description']}")

            # Show detailed execution results
            print(f"\nExecution details:")
            for step in result["execution_results"]:
                status = "✅" if step["success"] else "❌"
                print(f"  {status} Step {step['step']}: {step['description']}")
                if not step["success"]:
                    print(f"    Error: {step.get('error', 'Unknown error')}")

        print("\n🎉 Advanced AI research completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def complex_research_task():
    """Example of a complex multi-step research task."""
    print("\n🔬 Complex Research Task Example")
    print("=" * 40)

    complex_task = """
    Research the intersection of artificial intelligence and climate change solutions.
    Find recent papers, analyze trends, and identify promising research directions
    for AI applications in environmental sustainability.
    """

    try:
        result = await ai_research(complex_task)

        print(f"🎯 Complex task generated {len(result['plan'])} step plan:")
        for i, step in enumerate(result["plan"], 1):
            print(f"  {i}. {step['description']} (using {step['tool']})")

        summary = result["summary"]
        print(f"\n📊 Results:")
        print(f"  Success rate: {summary['successful_steps']}/{summary['total_steps']}")
        print(f"  Tools utilized: {', '.join(summary['tools_used'])}")

        print("\n🎉 Complex research task completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all AI research examples."""
    print("🚀 AI Research Examples")
    print("=" * 50)

    # Run examples in sequence
    await simple_ai_research()
    await advanced_ai_research()
    await complex_research_task()

    print("\n🏁 All AI research examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
