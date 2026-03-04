"""
Autonomous Research Agent
A LangChain + LangGraph powered agent that researches topics,
reads papers, and synthesizes findings into structured reports.
"""

from .agent import ResearchAgent, create_agent
from .graph import create_research_graph, get_initial_state, run_research
from .tools import SearchTool, PaperReaderTool, WebScraperTool, get_all_tools
from .interfaces import (
    BaseSearchTool,
    BaseConnector,
    BaseNode,
    SourceResult,
    ResearchFinding,
    RetryMixin,
    CacheMixin,
    RateLimitMixin,
)
from .prompts import (
    PLANNING_SYSTEM_PROMPT,
    EXTRACTION_PROMPT_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    get_planning_prompt,
    get_extraction_prompt,
    get_synthesis_prompt,
)

__all__ = [
    # Main classes
    "ResearchAgent",
    "create_agent",
    
    # Graph
    "create_research_graph",
    "get_initial_state",
    "run_research",
    
    # Tools
    "SearchTool",
    "PaperReaderTool", 
    "WebScraperTool",
    "get_all_tools",
    
    # Interfaces (for extending)
    "BaseSearchTool",
    "BaseConnector",
    "BaseNode",
    "SourceResult",
    "ResearchFinding",
    "RetryMixin",
    "CacheMixin",
    "RateLimitMixin",
    
    # Prompts (for customization)
    "PLANNING_SYSTEM_PROMPT",
    "EXTRACTION_PROMPT_TEMPLATE",
    "SYNTHESIS_PROMPT_TEMPLATE",
    "get_planning_prompt",
    "get_extraction_prompt",
    "get_synthesis_prompt",
]
