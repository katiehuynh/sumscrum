# 🔬 Autonomous Research Agent

An AI-powered research agent built with **LangChain** and **LangGraph** that autonomously:
- Searches the web for information on any topic
- Finds and reads academic papers from ArXiv
- Synthesizes findings into structured reports with proper citations

## Quick Start

### 1. Clone and Setup Environment

```bash
cd sumscrum
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit the `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

**Get your keys:**
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://tavily.com (free tier: 1000 searches/month)

### 3. Run the Agent

```bash
# Research a topic
python main.py "Impact of artificial intelligence on healthcare"

# Interactive mode
python main.py --interactive

# Save to specific file
python main.py "Climate change solutions" -o climate_report.md

# Use GPT-4 for better quality
python main.py "Quantum computing basics" --model gpt-4o
```

## 📁 Project Structure

```
sumscrum/
├── .env                 # API keys (not committed to git)
├── requirements.txt     # Python dependencies
├── main.py             # CLI entry point
├── src/
│   ├── __init__.py     # Package exports
│   ├── agent.py        # Main ResearchAgent class
│   ├── graph.py        # LangGraph workflow definition
│   ├── tools.py        # Research tools (search, read, etc.)
│   └── utils.py        # Helper utilities
└── tests/
    └── test_agent.py   # Unit tests
```

## 🔧 How It Works

The agent uses a **LangGraph** state machine with four main stages:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Planning   │────▶│  Searching  │────▶│   Reading   │────▶│ Synthesizing │
│             │     │             │     │             │     │              │
│ Break down  │     │ Web search  │     │ Extract key │     │ Generate     │
│ into topics │     │ ArXiv papers│     │ findings    │     │ report       │
└─────────────┘     └─────────────┘     └─────────────┘     └──────────────┘
```

1. **Planning**: Breaks down the research topic into specific subtopics
2. **Searching**: Uses Tavily API and ArXiv to find relevant sources
3. **Reading**: Extracts and processes content from sources
4. **Synthesizing**: Generates a structured report with citations

## Usage Examples

### Python API

```python
from src.agent import ResearchAgent

# Create agent
agent = ResearchAgent(model="gpt-4o-mini", verbose=True)

# Run full research
report = agent.research("Machine learning in drug discovery")
print(report)

# Quick web search
results = agent.quick_search("latest AI breakthroughs 2024")

# Search academic papers
papers = agent.search_papers("transformer architecture")

# Save report to file
agent.save_report(report, "my_research.md")
```

### Interactive Mode Commands

- `research <topic>` - Full research on a topic
- `search <query>` - Quick web search
- `papers <query>` - Search ArXiv papers
- `ask <question>` - Ask a general question
- `quit` - Exit

## 🛠️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM |
| `TAVILY_API_KEY` | Recommended | Tavily API key for web search |

### Model Options

- `gpt-4o-mini` (default) - Fast and cost-effective
- `gpt-4o` - Higher quality, more expensive
- `gpt-4-turbo` - Good balance of quality and speed

## Output Format

Reports are generated in Markdown format with:

- **Executive Summary** - Brief overview
- **Key Findings** - Organized by subtopic with citations
- **Analysis** - Insights and trends
- **Conclusions** - Main takeaways
- **References** - Numbered source list

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request