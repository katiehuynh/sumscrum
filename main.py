#!/usr/bin/env python3
"""
Autonomous Research Agent - Main Entry Point

A LangChain + LangGraph powered agent that:
- Searches the web for information
- Reads academic papers from ArXiv
- Synthesizes findings into structured reports with citations

Usage:
    python main.py "Your research topic here"
    python main.py --interactive
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_environment():
    """Check that required environment variables are set."""
    errors = []
    
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is not set")
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY is not set. Web search will be limited.")
    
    if errors:
        print("\n Environment Setup Errors:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease set these in your .env file and try again.")
        sys.exit(1)


def run_research(topic: str, output_file: str = None, model: str = "gpt-4o-mini"):
    """
    Run research on a topic and optionally save to file.
    
    Args:
        topic: The research topic
        output_file: Optional output file path
        model: OpenAI model to use
    """
    from src.agent import ResearchAgent
    
    print("\n" + "=" * 60)
    print("AUTONOMOUS RESEARCH AGENT")
    print("=" * 60)
    
    # Create agent
    agent = ResearchAgent(model=model, verbose=True)
    
    # Run research
    report = agent.research(topic)
    
    # Output report
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60 + "\n")
    print(report)
    
    # Save to file if specified
    if output_file:
        agent.save_report(report, output_file)
    else:
        # Auto-generate filename
        from src.utils import generate_report_filename
        filename = generate_report_filename(topic)
        agent.save_report(report, filename)
    
    return report


def interactive_mode():
    """Run the agent in interactive mode."""
    from src.agent import ResearchAgent
    
    print("\n" + "=" * 60)
    print("AUTONOMOUS RESEARCH AGENT - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  research <topic>  - Research a topic")
    print("  search <query>    - Quick web search")
    print("  papers <query>    - Search academic papers")
    print("  ask <question>    - Ask a question")
    print("  help              - Show this help")
    print("  quit              - Exit")
    print()
    
    agent = ResearchAgent(verbose=True)
    
    while True:
        try:
            user_input = input("\n Enter command: ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            elif command == "help":
                print("\nCommands:")
                print("  research <topic>  - Full research on a topic")
                print("  search <query>    - Quick web search")
                print("  papers <query>    - Search ArXiv papers")
                print("  ask <question>    - Ask a question")
                print("  quit              - Exit")
            
            elif command == "research":
                if not args:
                    print("Please provide a research topic.")
                else:
                    report = agent.research(args)
                    print("\n" + report)
            
            elif command == "search":
                if not args:
                    print("Please provide a search query.")
                else:
                    results = agent.quick_search(args)
                    print("\n Search Results:")
                    for i, r in enumerate(results, 1):
                        print(f"\n{i}. {r.get('title', 'No title')}")
                        print(f"   URL: {r.get('url', 'No URL')}")
                        print(f"   {r.get('content', 'No content')[:200]}...")
            
            elif command == "papers":
                if not args:
                    print("Please provide a search query.")
                else:
                    papers = agent.search_papers(args)
                    print("\n Academic Papers:")
                    for i, p in enumerate(papers, 1):
                        print(f"\n{i}. {p.get('title', 'No title')}")
                        if p.get('authors'):
                            print(f"   Authors: {', '.join(p['authors'][:3])}")
                        print(f"   URL: {p.get('url', 'No URL')}")
                        print(f"   Abstract: {p.get('abstract', 'No abstract')[:200]}...")
            
            elif command == "ask":
                if not args:
                    print("Please provide a question.")
                else:
                    answer = agent.ask(args)
                    print(f"\n {answer}")
            
            else:
                # Treat the whole input as a research topic
                print(f"\nStarting research on: {user_input}")
                report = agent.research(user_input)
                print("\n" + report)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent - Research topics with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Impact of AI on healthcare"
  python main.py "Climate change mitigation strategies" -o climate_report.md
  python main.py --interactive
  python main.py "Quantum computing" --model gpt-4o
        """
    )
    
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic (optional if using --interactive)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path for the report"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    
    args = parser.parse_args()
    
    # Check environment
    check_environment()
    
    if args.interactive:
        interactive_mode()
    elif args.topic:
        run_research(args.topic, args.output, args.model)
    else:
        parser.print_help()
        print("\nTip: Use --interactive for interactive mode, or provide a topic.")
        sys.exit(1)


if __name__ == "__main__":
    main()
