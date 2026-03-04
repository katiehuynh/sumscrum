"""
Main Research Agent class that orchestrates the research workflow.
Provides a high-level interface for running research tasks.
"""

import os
from typing import Optional, Dict, Any, List, Callable
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from .graph import create_research_graph, get_initial_state, run_research
from .tools import get_all_tools


class ResearchAgent:
    """
    Autonomous Research Agent that searches the web, reads papers,
    and synthesizes findings into structured reports with citations.
    
    Usage:
        agent = ResearchAgent()
        report = agent.research("Impact of AI on healthcare")
        print(report)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        verbose: bool = True,
        max_sources: int = 10,
    ):
        """
        Initialize the Research Agent.
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            temperature: LLM temperature (default: 0 for consistency)
            verbose: Whether to print progress updates
            max_sources: Maximum number of sources to process
        """
        load_dotenv()
        
        self._validate_api_keys()
        
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.max_sources = max_sources
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        
        # Create the research graph
        self.graph = create_research_graph(self.llm)
        
        # Get available tools
        self.tools = get_all_tools()
        
        if self.verbose:
            print(f" Research Agent initialized with {model}")
    
    def _validate_api_keys(self) -> None:
        """Validate that required API keys are set."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file."
            )
        
        if not os.getenv("TAVILY_API_KEY"):
            print("Warning: TAVILY_API_KEY not found. Web search will be limited.")
    
    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def research(
        self,
        topic: str,
        callback: Optional[Callable[[str, Any], None]] = None,
    ) -> str:
        """
        Conduct research on a given topic.
        
        Args:
            topic: The research topic or question
            callback: Optional callback function for progress updates
                     Called with (step_name, state) on each step
        
        Returns:
            A structured research report with citations
        """
        self._log(f"\n Starting research on: {topic}")
        self._log("=" * 50)
        
        # Initialize state
        initial_state = get_initial_state(topic)
        
        # Run the research graph with streaming to track progress
        try:
            for step_output in self.graph.stream(initial_state):
                step_name = list(step_output.keys())[0]
                step_state = step_output[step_name]
                
                self._log(f"\n Step: {step_name}")
                
                if step_name == "planning":
                    subtopics = step_state.get("subtopics", [])
                    self._log(f"   Identified {len(subtopics)} subtopics:")
                    for st in subtopics:
                        self._log(f"   - {st}")
                
                elif step_name == "searching":
                    sources = step_state.get("sources", [])
                    self._log(f"   Found {len(sources)} sources")
                
                elif step_name == "reading":
                    findings = step_state.get("findings", [])
                    self._log(f"   Extracted findings from {len(findings)} sources")
                
                elif step_name == "synthesizing":
                    self._log("   Generating final report...")
                
                # Call user callback if provided
                if callback:
                    callback(step_name, step_state)
                
                # Store final state
                final_state = step_state
            
            self._log("\n Research complete!")
            self._log("=" * 50)
            
            return final_state.get("report", "No report generated.")
        
        except Exception as e:
            self._log(f"\n Error during research: {str(e)}")
            raise
    
    def quick_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a quick web search without full research workflow.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        from .tools import search_web
        return search_web.invoke({"query": query, "max_results": 5})
    
    def search_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for academic papers on ArXiv.
        
        Args:
            query: Search query for papers
            max_results: Maximum number of results
            
        Returns:
            List of paper information
        """
        from .tools import search_arxiv
        return search_arxiv.invoke({"query": query, "max_results": max_results})
    
    def read_url(self, url: str) -> Dict[str, Any]:
        """
        Read content from a specific URL.
        
        Args:
            url: URL to read
            
        Returns:
            Dictionary with title and content
        """
        from .tools import read_webpage
        return read_webpage.invoke({"url": url})
    
    def ask(self, question: str, context: str = "") -> str:
        """
        Ask a question with optional context.
        
        Args:
            question: The question to ask
            context: Optional context to include
            
        Returns:
            LLM response
        """
        messages = []
        
        if context:
            messages.append(SystemMessage(content=f"Context:\n{context}"))
        
        messages.append(HumanMessage(content=question))
        
        response = self.llm.invoke(messages)
        return response.content
    
    def save_report(self, report: str, filename: str = "research_report.md") -> str:
        """
        Save the research report to a file.
        
        Args:
            report: The report content
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        self._log(f" Report saved to: {filename}")
        return filename


def create_agent(
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> ResearchAgent:
    """
    Factory function to create a Research Agent.
    
    Args:
        model: OpenAI model to use
        verbose: Whether to print progress
        
    Returns:
        Configured ResearchAgent instance
    """
    return ResearchAgent(model=model, verbose=verbose)
