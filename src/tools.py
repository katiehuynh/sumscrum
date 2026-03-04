"""
Research Tools for the Autonomous Research Agent.
Defines tools for web search, paper reading, and content extraction.
"""

import os
import re
import arxiv
import requests
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Schema for search results."""
    title: str
    url: str
    content: str
    score: Optional[float] = None


class PaperInfo(BaseModel):
    """Schema for academic paper information."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    published: Optional[str] = None
    source: str = "arxiv"


# ============================================================
# Web Search Tools
# ============================================================

def get_tavily_search_tool(max_results: int = 5) -> TavilySearchResults:
    """
    Create a Tavily search tool for web searches.
    
    Args:
        max_results: Maximum number of results to return
        
    Returns:
        Configured TavilySearchResults tool
    """
    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
    )


@tool
def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for information on a given query using Tavily.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and content
    """
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )
        
        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0),
            })
        
        return results
    except Exception as e:
        return [{"error": str(e), "title": "", "url": "", "content": ""}]


# ============================================================
# Academic Paper Tools
# ============================================================

@tool
def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search ArXiv for academic papers on a given topic.
    
    Args:
        query: The search query for academic papers
        max_results: Maximum number of papers to return
        
    Returns:
        List of papers with title, authors, abstract, and URLs
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "published": str(result.published.date()) if result.published else None,
                "source": "arxiv"
            })
        
        return papers
    except Exception as e:
        return [{"error": str(e)}]


@tool
def read_webpage(url: str) -> Dict[str, Any]:
    """
    Read and extract the main content from a webpage.
    
    Args:
        url: The URL of the webpage to read
        
    Returns:
        Dictionary with title and extracted content
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Get title
        title = soup.title.string if soup.title else "No title"
        
        # Get main content - try common content containers
        main_content = None
        for selector in ["article", "main", ".content", ".post", "#content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body
        
        # Extract text
        text = main_content.get_text(separator="\n", strip=True) if main_content else ""
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text[:8000]  # Limit content length
        
        return {
            "title": title,
            "url": url,
            "content": text,
            "success": True
        }
    except Exception as e:
        return {
            "title": "",
            "url": url,
            "content": f"Error reading webpage: {str(e)}",
            "success": False
        }


@tool
def extract_key_points(text: str, topic: str) -> str:
    """
    Extract key points from text relevant to a specific topic.
    This is a helper tool that marks text for LLM processing.
    
    Args:
        text: The text to extract key points from
        topic: The topic to focus on when extracting points
        
    Returns:
        Instruction string for LLM processing
    """
    return f"""
    Please extract the key points from the following text that are relevant to: {topic}
    
    Text:
    {text[:4000]}
    
    Provide a bullet-point summary of the most important findings.
    """


# ============================================================
# Tool Collection
# ============================================================

def get_all_tools():
    """
    Get all available research tools.
    
    Returns:
        List of tool functions
    """
    return [
        search_web,
        search_arxiv,
        read_webpage,
        extract_key_points,
    ]


class SearchTool:
    """Wrapper class for search tools."""
    
    @staticmethod
    def web_search(query: str, max_results: int = 5) -> List[Dict]:
        return search_web.invoke({"query": query, "max_results": max_results})
    
    @staticmethod
    def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
        return search_arxiv.invoke({"query": query, "max_results": max_results})


class PaperReaderTool:
    """Wrapper class for paper reading tools."""
    
    @staticmethod
    def read_paper(url: str) -> Dict:
        return read_webpage.invoke({"url": url})


class WebScraperTool:
    """Wrapper class for web scraping tools."""
    
    @staticmethod
    def scrape(url: str) -> Dict:
        return read_webpage.invoke({"url": url})
