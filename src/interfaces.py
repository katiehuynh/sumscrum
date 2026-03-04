"""
Abstract Base Classes and Protocols for the Research Agent.

Defines interfaces for tools, connectors, and nodes to ensure
consistent behavior and enable easy extension.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from pydantic import BaseModel


# ============================================================
# Data Models
# ============================================================

class SourceResult(BaseModel):
    """Standard result format for all data sources."""
    title: str
    url: str
    content: str
    source_type: str
    subtopic: str = ""
    metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ResearchFinding(BaseModel):
    """A finding extracted from a source."""
    source_title: str
    source_url: str
    subtopic: str
    key_points: str
    source_type: str = "web"


# ============================================================
# Tool Protocol
# ============================================================

@runtime_checkable
class SearchToolProtocol(Protocol):
    """Protocol for search tools."""
    
    def invoke(self, input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the search and return results."""
        ...


@runtime_checkable  
class ReaderToolProtocol(Protocol):
    """Protocol for content reader tools."""
    
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Read content and return extracted data."""
        ...


# ============================================================
# Abstract Base Classes
# ============================================================

class BaseSearchTool(ABC):
    """Abstract base class for search tools."""
    
    source_type: str = "unknown"
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[SourceResult]:
        """
        Search for information.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
            
        Returns:
            List of SourceResult objects
        """
        pass
    
    def invoke(self, input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LangChain-compatible invoke method."""
        results = self.search(input.get("query", ""), **input)
        return [r.to_dict() for r in results]


class BaseConnector(ABC):
    """Abstract base class for enterprise connectors."""
    
    source_type: str = "enterprise"
    
    def __init__(self, **config):
        """Initialize with configuration."""
        self.config = config
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[SourceResult]:
        """Search the data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""
        pass
    
    def invoke(self, input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LangChain-compatible invoke method."""
        try:
            self.connect()
            results = self.search(input.get("query", ""), **input)
            return [r.to_dict() for r in results]
        finally:
            self.disconnect()


class BaseNode(ABC):
    """Abstract base class for graph nodes."""
    
    name: str = "base_node"
    
    def __init__(self, llm=None, **config):
        """Initialize node with optional LLM and config."""
        self.llm = llm
        self.config = config
    
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the state and return updated state.
        
        Args:
            state: Current research state
            
        Returns:
            Updated research state
        """
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the node callable for LangGraph."""
        return self.process(state)


# ============================================================
# Mixins
# ============================================================

class RetryMixin:
    """Mixin for adding retry logic to tools."""
    
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        import time
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise last_error


class CacheMixin:
    """Mixin for adding caching to tools."""
    
    _cache: Dict[str, Any] = {}
    cache_ttl: int = 300  # seconds
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        import time
        
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return value
            del self._cache[key]
        return None
    
    def set_cached(self, key: str, value: Any) -> None:
        """Set item in cache with timestamp."""
        import time
        self._cache[key] = (value, time.time())


class RateLimitMixin:
    """Mixin for rate limiting API calls."""
    
    calls_per_minute: int = 60
    _call_times: List[float] = []
    
    def check_rate_limit(self) -> None:
        """Check and enforce rate limit."""
        import time
        
        now = time.time()
        # Remove calls older than 1 minute
        self._call_times = [t for t in self._call_times if now - t < 60]
        
        if len(self._call_times) >= self.calls_per_minute:
            sleep_time = 60 - (now - self._call_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._call_times.append(now)


# ============================================================
# Type Aliases
# ============================================================

ResearchState = Dict[str, Any]
ToolResult = List[Dict[str, Any]]
NodeFunction = callable
