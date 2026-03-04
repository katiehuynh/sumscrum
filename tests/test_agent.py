"""
Tests for the Autonomous Research Agent.
"""

import pytest
from unittest.mock import patch, MagicMock
import os


# Set dummy API keys for testing
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["TAVILY_API_KEY"] = "test-tavily-key"


class TestUtils:
    """Test utility functions."""
    
    def test_clean_text(self):
        from src.utils import clean_text
        
        # Test whitespace cleaning
        assert clean_text("  hello   world  ") == "hello world"
        
        # Test empty string
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_truncate_text(self):
        from src.utils import truncate_text
        
        # Text shorter than max length
        assert truncate_text("hello", 10) == "hello"
        
        # Text longer than max length
        result = truncate_text("hello world", 8)
        assert len(result) == 8
        assert result.endswith("...")
    
    def test_format_citation(self):
        from src.utils import format_citation
        
        citation = format_citation(
            title="Test Paper",
            url="https://example.com",
            authors=["John Doe", "Jane Smith"],
            date="2024"
        )
        
        assert "Test Paper" in citation
        assert "https://example.com" in citation
        assert "John Doe" in citation
    
    def test_generate_report_filename(self):
        from src.utils import generate_report_filename
        
        filename = generate_report_filename("AI in Healthcare")
        
        assert filename.startswith("research_")
        assert filename.endswith(".md")
        assert "ai" in filename.lower()
    
    def test_parse_json_safely(self):
        from src.utils import parse_json_safely
        
        # Valid JSON
        assert parse_json_safely('["a", "b"]') == ["a", "b"]
        
        # JSON in markdown
        result = parse_json_safely('```json\n["a", "b"]\n```')
        assert result == ["a", "b"]
        
        # Invalid JSON
        assert parse_json_safely('not json', default=[]) == []
    
    def test_create_markdown_table(self):
        from src.utils import create_markdown_table
        
        table = create_markdown_table(
            headers=["Name", "Value"],
            rows=[["A", "1"], ["B", "2"]]
        )
        
        assert "| Name | Value |" in table
        assert "| A | 1 |" in table


class TestTools:
    """Test research tools."""
    
    def test_tool_imports(self):
        """Test that all tools can be imported."""
        from src.tools import (
            search_web,
            search_arxiv,
            read_webpage,
            extract_key_points,
            get_all_tools,
        )
        
        tools = get_all_tools()
        assert len(tools) == 4
    
    @patch('src.tools.requests.get')
    def test_read_webpage(self, mock_get):
        """Test webpage reading with mocked response."""
        from src.tools import read_webpage
        
        # Mock response
        mock_response = MagicMock()
        mock_response.text = '<html><head><title>Test</title></head><body><p>Content</p></body></html>'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = read_webpage.invoke({"url": "https://example.com"})
        
        assert result["success"] == True
        assert result["title"] == "Test"


class TestGraph:
    """Test the research graph."""
    
    def test_initial_state(self):
        from src.graph import get_initial_state
        
        state = get_initial_state("Test topic")
        
        assert state["topic"] == "Test topic"
        assert state["subtopics"] == []
        assert state["sources"] == []
        assert state["current_step"] == "planning"
    
    @patch('src.graph.ChatOpenAI')
    def test_create_research_graph(self, mock_llm):
        from src.graph import create_research_graph
        
        # Should not raise
        graph = create_research_graph(mock_llm)
        assert graph is not None


class TestAgent:
    """Test the research agent."""
    
    @patch('src.agent.ChatOpenAI')
    @patch('src.agent.create_research_graph')
    def test_agent_creation(self, mock_graph, mock_llm):
        """Test that agent can be created."""
        from src.agent import ResearchAgent
        
        agent = ResearchAgent(verbose=False)
        assert agent is not None
        assert agent.model == "gpt-4o-mini"
    
    @patch('src.agent.ChatOpenAI')
    @patch('src.agent.create_research_graph')
    def test_quick_search(self, mock_graph, mock_llm):
        """Test quick search method."""
        from src.agent import ResearchAgent
        
        with patch('src.agent.get_all_tools'):
            agent = ResearchAgent(verbose=False)
            
            with patch('src.tools.search_web') as mock_search:
                mock_search.invoke.return_value = [{"title": "Test", "url": "http://test.com"}]
                # Would test actual behavior here


# Run tests with: pytest tests/test_agent.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
