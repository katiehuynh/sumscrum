"""
Utility functions for the Autonomous Research Agent.
"""

import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_citation(
    title: str,
    url: str,
    authors: Optional[List[str]] = None,
    date: Optional[str] = None,
    source_type: str = "web"
) -> str:
    """
    Format a citation in a consistent style.
    
    Args:
        title: Source title
        url: Source URL
        authors: List of authors (optional)
        date: Publication date (optional)
        source_type: Type of source (web, arxiv, etc.)
        
    Returns:
        Formatted citation string
    """
    parts = []
    
    if authors:
        if len(authors) > 3:
            parts.append(f"{authors[0]} et al.")
        else:
            parts.append(", ".join(authors))
    
    if date:
        parts.append(f"({date})")
    
    parts.append(f'"{title}"')
    
    if source_type == "arxiv":
        parts.append("arXiv")
    
    parts.append(url)
    
    return ". ".join(parts)


def format_report_section(
    title: str,
    content: str,
    level: int = 2
) -> str:
    """
    Format a report section with proper markdown.
    
    Args:
        title: Section title
        content: Section content
        level: Header level (1-6)
        
    Returns:
        Formatted markdown section
    """
    header = "#" * level
    return f"{header} {title}\n\n{content}\n\n"


def parse_json_safely(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON from text that might contain markdown or other content.
    
    Args:
        text: Text that might contain JSON
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        # Try direct parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\[[\s\S]*\]',
        r'\{[\s\S]*\}',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1) if '```' in pattern else match.group())
            except json.JSONDecodeError:
                continue
    
    return default


def create_markdown_table(
    headers: List[str],
    rows: List[List[str]]
) -> str:
    """
    Create a markdown table from headers and rows.
    
    Args:
        headers: Table headers
        rows: Table rows (list of lists)
        
    Returns:
        Markdown table string
    """
    if not headers or not rows:
        return ""
    
    # Header row
    header_row = "| " + " | ".join(headers) + " |"
    
    # Separator row
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    # Data rows
    data_rows = []
    for row in rows:
        # Ensure row has correct number of columns
        while len(row) < len(headers):
            row.append("")
        data_rows.append("| " + " | ".join(row[:len(headers)]) + " |")
    
    return "\n".join([header_row, separator] + data_rows)


def generate_report_filename(topic: str, extension: str = "md") -> str:
    """
    Generate a filename for a research report.
    
    Args:
        topic: Research topic
        extension: File extension
        
    Returns:
        Generated filename
    """
    # Clean topic for filename
    clean_topic = re.sub(r'[^\w\s-]', '', topic.lower())
    clean_topic = re.sub(r'[-\s]+', '_', clean_topic)
    clean_topic = clean_topic[:50]  # Limit length
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"research_{clean_topic}_{timestamp}.{extension}"


def merge_findings(findings: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Merge findings from multiple sources by topic.
    
    Args:
        findings: List of finding dictionaries
        
    Returns:
        Merged findings organized by subtopic
    """
    merged = {}
    
    for finding in findings:
        subtopic = finding.get("subtopic", "general")
        key_points = finding.get("key_points", "")
        
        if subtopic not in merged:
            merged[subtopic] = []
        
        # Split key points if they're a string
        if isinstance(key_points, str):
            points = [
                p.strip().lstrip("-•").strip() 
                for p in key_points.split("\n") 
                if p.strip()
            ]
            merged[subtopic].extend(points)
        elif isinstance(key_points, list):
            merged[subtopic].extend(key_points)
    
    return merged


def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """
    Estimate reading time for text in minutes.
    
    Args:
        text: Text content
        wpm: Words per minute (default: 200)
        
    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return max(1, round(word_count / wpm))


class ResearchCache:
    """Simple in-memory cache for research results."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
