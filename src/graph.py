"""
LangGraph Workflow for the Autonomous Research Agent.
Defines the state machine and workflow nodes for research tasks.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import operator


# ============================================================
# State Definitions
# ============================================================

class Source(BaseModel):
    """A source with citation information."""
    title: str
    url: str
    content_summary: str
    source_type: str = "web"  # web, arxiv, pdf


class ResearchState(TypedDict):
    """
    The state of the research agent at any point in the workflow.
    
    Attributes:
        topic: The main research topic
        messages: Conversation history
        subtopics: List of subtopics to research
        sources: Collected sources with citations
        findings: Key findings from research
        current_step: Current step in the workflow
        report: Final synthesized report
        error: Any error message
    """
    topic: str
    messages: Annotated[List[BaseMessage], add_messages]
    subtopics: List[str]
    sources: List[Dict[str, Any]]
    findings: List[str]
    current_step: str
    report: str
    error: Optional[str]


def get_initial_state(topic: str) -> ResearchState:
    """Create initial state for a research task."""
    return {
        "topic": topic,
        "messages": [],
        "subtopics": [],
        "sources": [],
        "findings": [],
        "current_step": "planning",
        "report": "",
        "error": None,
    }


# ============================================================
# Node Functions
# ============================================================

def create_planning_node(llm: ChatOpenAI):
    """
    Create a planning node that breaks down the research topic into
    educational subtopics that build understanding progressively.
    
    Args:
        llm: The language model to use
        
    Returns:
        Node function
    """
    def planning_node(state: ResearchState) -> ResearchState:
        """Break down the research topic into progressive subtopics."""
        topic = state["topic"]
        
        system_prompt = """You are an educational research planner. Given a research topic, 
        break it down into 5-7 subtopics that will help someone with ZERO prior knowledge 
        fully understand the topic.
        
        Structure the subtopics to build understanding progressively:
        1. First subtopics should cover foundational concepts and definitions
        2. Middle subtopics should cover how things work and key mechanisms
        3. Later subtopics should cover applications, challenges, and future trends
        
        Each subtopic should be specific enough to search for, but broad enough to find good sources.
        
        Return ONLY a JSON list of strings in learning order, like:
        ["foundational concept 1", "how X works", "applications of X", "challenges in X", "future of X"]
        
        Do not include any other text."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research topic: {topic}")
        ]
        
        response = llm.invoke(messages)
        
        # Parse subtopics from response
        try:
            import json
            content = response.content.strip()
            # Handle markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            subtopics = json.loads(content)
        except:
            # Fallback: split by newlines
            subtopics = [
                line.strip().strip("-").strip("•").strip() 
                for line in response.content.split("\n") 
                if line.strip() and not line.startswith("{")
            ][:7]
        
        return {
            **state,
            "subtopics": subtopics,
            "current_step": "searching",
            "messages": state["messages"] + [
                HumanMessage(content=f"Research topic: {topic}"),
                AIMessage(content=f"I've identified these subtopics to research: {subtopics}")
            ]
        }
    
    return planning_node


def create_search_node(llm: ChatOpenAI, enterprise_connectors: List[str] = None):
    """
    Create a search node that finds diverse sources including beginner-friendly content.
    Supports optional enterprise connectors for internal data sources.
    
    Args:
        llm: The language model to use
        enterprise_connectors: List of connector names to use (e.g., ['confluence', 'slack'])
        
    Returns:
        Node function
    """
    def search_node(state: ResearchState) -> ResearchState:
        """Search for sources on each subtopic, including introductory content."""
        from .tools import search_web, search_arxiv
        
        all_sources = []
        topic = state["topic"]
        
        # ============================================================
        # Enterprise Connectors (if configured)
        # ============================================================
        if enterprise_connectors:
            try:
                from .connectors import get_connector_by_name
                
                for connector_name in enterprise_connectors:
                    connector = get_connector_by_name(connector_name)
                    if connector:
                        try:
                            results = connector.invoke({"query": topic})
                            for result in results:
                                if "error" not in result:
                                    all_sources.append({
                                        **result,
                                        "subtopic": "internal knowledge"
                                    })
                            print(f"✓ {connector_name}: Found {len(results)} results")
                        except Exception as e:
                            print(f"Enterprise connector '{connector_name}' error: {e}")
            except ImportError:
                print("Enterprise connectors not available")
        
        # ============================================================
        # Web Search (Default)
        # ============================================================
        # First, search for introductory/overview content
        try:
            intro_results = search_web.invoke({
                "query": f"what is {topic} explained simply introduction basics",
                "max_results": 3
            })
            for result in intro_results:
                if "error" not in result:
                    all_sources.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "source_type": "web",
                        "subtopic": "introduction and basics"
                    })
        except Exception as e:
            print(f"Intro search error: {e}")
        
        # Search for each subtopic
        for subtopic in state["subtopics"]:
            # Web search - regular query
            try:
                web_results = search_web.invoke({
                    "query": f"{topic} {subtopic}",
                    "max_results": 3
                })
                for result in web_results:
                    if "error" not in result:
                        all_sources.append({
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "content": result.get("content", ""),
                            "source_type": "web",
                            "subtopic": subtopic
                        })
            except Exception as e:
                print(f"Web search error for '{subtopic}': {e}")
            
            # ArXiv search for academic papers (for deeper content)
            try:
                arxiv_results = search_arxiv.invoke({
                    "query": f"{topic} {subtopic}",
                    "max_results": 2
                })
                for paper in arxiv_results:
                    if "error" not in paper:
                        all_sources.append({
                            "title": paper.get("title", ""),
                            "url": paper.get("url", ""),
                            "content": paper.get("abstract", ""),
                            "authors": paper.get("authors", []),
                            "source_type": "arxiv",
                            "subtopic": subtopic
                        })
            except Exception as e:
                print(f"ArXiv search error for '{subtopic}': {e}")
        
        # Search for examples and applications
        try:
            examples_results = search_web.invoke({
                "query": f"{topic} real world examples applications use cases",
                "max_results": 2
            })
            for result in examples_results:
                if "error" not in result:
                    all_sources.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "source_type": "web",
                        "subtopic": "examples and applications"
                    })
        except Exception as e:
            print(f"Examples search error: {e}")
        
        return {
            **state,
            "sources": all_sources,
            "current_step": "reading",
            "messages": state["messages"] + [
                AIMessage(content=f"Found {len(all_sources)} sources to analyze.")
            ]
        }
    
    return search_node


def create_reading_node(llm: ChatOpenAI):
    """
    Create a reading node that extracts educational content from sources.
    
    Args:
        llm: The language model to use
        
    Returns:
        Node function
    """
    def reading_node(state: ResearchState) -> ResearchState:
        """Read and extract educational findings from sources."""
        from .tools import read_webpage
        
        findings = []
        enhanced_sources = []
        
        for source in state["sources"][:12]:  # Process up to 12 sources
            # For web sources, try to get more content
            if source["source_type"] == "web" and source.get("url"):
                try:
                    full_content = read_webpage.invoke({"url": source["url"]})
                    if full_content.get("success"):
                        source["full_content"] = full_content.get("content", "")[:4000]
                except:
                    pass
            
            # Extract educational content using LLM
            content = source.get("full_content", source.get("content", ""))
            subtopic = source.get("subtopic", state["topic"])
            
            if content:
                extract_prompt = f"""You are extracting educational content about "{state['topic']}" 
(specifically related to: {subtopic}).

Source Title: {source.get('title', 'Unknown')}
Source Content: {content[:3000]}

Extract the following (if present in the source):

## Key Facts
- List 2-4 important factual findings (start each with "- ")

## Definitions  
- Any important terms defined in this source

## How It Works
- Any explanations of mechanisms, processes, or how things function

## Examples
- Any real-world examples, case studies, or applications mentioned

## Important Numbers
- Any statistics, dates, or quantitative data

Be concise but informative. Only include sections where you found relevant content."""

                try:
                    response = llm.invoke([HumanMessage(content=extract_prompt)])
                    findings.append({
                        "source_title": source.get("title", ""),
                        "source_url": source.get("url", ""),
                        "subtopic": subtopic,
                        "key_points": response.content,
                        "source_type": source.get("source_type", "web")
                    })
                except Exception as e:
                    print(f"Error extracting from source: {e}")
            
            enhanced_sources.append(source)
        
        return {
            **state,
            "sources": enhanced_sources,
            "findings": findings,
            "current_step": "synthesizing",
            "messages": state["messages"] + [
                AIMessage(content=f"Extracted findings from {len(findings)} sources.")
            ]
        }
    
    return reading_node


def create_synthesis_node(llm: ChatOpenAI):
    """
    Create a synthesis node that generates an educational, progressively-structured report.
    
    The report is designed for readers completely new to the subject, starting with
    foundational concepts and building to deeper understanding.
    
    Args:
        llm: The language model to use
        
    Returns:
        Node function
    """
    def synthesis_node(state: ResearchState) -> ResearchState:
        """Synthesize findings into a structured educational report with citations."""
        
        # Prepare findings summary
        findings_text = "\n\n".join([
            f"### Source: {f['source_title']}\nURL: {f['source_url']}\n{f['key_points']}"
            for f in state["findings"]
        ])
        
        # Prepare source list for citations
        sources_list = "\n".join([
            f"[{i+1}] {s.get('title', 'Unknown')} - {s.get('url', 'No URL')}"
            for i, s in enumerate(state["sources"][:15])
        ])
        
        # Get subtopics for structure
        subtopics = state.get("subtopics", [])
        subtopics_text = "\n".join([f"- {st}" for st in subtopics])
        
        synthesis_prompt = f"""You are an expert educational writer and research synthesizer. Create a comprehensive, 
beginner-friendly research report on: "{state['topic']}"

Your goal is to make this topic FULLY UNDERSTANDABLE to someone with ZERO prior knowledge.
Structure the report to progressively build understanding from simple to complex.

## Research Findings Collected:
{findings_text}

## Subtopics Researched:
{subtopics_text}

## Available Sources for Citations:
{sources_list}

---

## REPORT STRUCTURE (Follow this exactly):

# [Topic Title]

## 1. The One-Sentence Summary
Start with a single, clear sentence that captures what this topic is about in the simplest terms possible.

## 2. Why Should You Care?
- What problem does this solve or address?  
- Who is affected by this topic?
- Why is it relevant right now?
(2-3 short paragraphs, written for someone who knows nothing)

## 3. The Big Picture: Core Concepts
Explain the 3-5 foundational ideas someone must understand FIRST before anything else makes sense.
Use this format for each concept:

### Concept 1: [Name]
**In plain English:** [One sentence explanation a 10-year-old could understand]
**The key insight:** [What makes this important]
**Analogy:** [Compare to something familiar from everyday life]

(Repeat for each core concept)

## 4. How It Works: The Building Blocks
Now that the reader understands the basics, explain how the pieces fit together.
- Use numbered steps or a logical flow
- Include a simple mental model or framework
- Explain cause and effect relationships

## 5. Deep Dive: Key Findings
Organize the research findings by subtopic. For each:

### [Subtopic Name]
**What we found:** [Key discoveries with citations like [1], [2]]
**What this means:** [Interpretation for a beginner]
**Current state:** [Where things stand today]

## 6. Real-World Applications
- Concrete examples of this topic in action
- Who is using this and how
- Impact on everyday life

## 7. Challenges & Limitations
- What problems or barriers exist
- What we don't know yet
- Common misconceptions to avoid

## 8. The Future: What's Next
- Emerging trends
- Open questions being researched
- Potential developments to watch

## 9. Key Takeaways
Provide 5-7 bullet points summarizing the most important things to remember.
Write these as "sticky" insights that are easy to recall.

## 10. Glossary
Define 8-12 key terms used in this report. Format:
- **Term**: Definition in simple language

## 11. Learn More
Suggest logical next steps for someone who wants to go deeper:
- What to learn next
- Related topics to explore
- Types of resources to seek out

## 12. References
List all sources used with full citations in this format:
[1] Author/Source. "Title." URL

---

## WRITING GUIDELINES:
- Write at an 8th-grade reading level for maximum accessibility
- Use short paragraphs (2-4 sentences max)
- Prefer active voice over passive
- Include analogies and comparisons to familiar things
- Define jargon immediately when first used
- Use bullet points and numbered lists liberally
- Bold key terms when introducing them
- Every section should flow logically to the next
- Assume the reader is intelligent but uninformed
- Cite sources using [number] notation throughout"""

        try:
            response = llm.invoke([HumanMessage(content=synthesis_prompt)])
            report = response.content
        except Exception as e:
            report = f"Error generating report: {str(e)}"
        
        return {
            **state,
            "report": report,
            "current_step": "complete",
            "messages": state["messages"] + [
                AIMessage(content="Research report completed.")
            ]
        }
    
    return synthesis_node


def create_error_node():
    """Create an error handling node."""
    def error_node(state: ResearchState) -> ResearchState:
        return {
            **state,
            "current_step": "error",
            "report": f"An error occurred: {state.get('error', 'Unknown error')}"
        }
    return error_node


# ============================================================
# Graph Construction
# ============================================================

def create_research_graph(
    llm: Optional[ChatOpenAI] = None,
    enterprise_connectors: List[str] = None
) -> StateGraph:
    """
    Create the research workflow graph.
    
    Args:
        llm: Optional language model (creates default if not provided)
        enterprise_connectors: List of enterprise connector names to use
                              Options: 'confluence', 'sharepoint', 'notion', 
                                      'slack', 'postgresql', 'elasticsearch',
                                      's3', 'azure', 'pinecone', 'chromadb'
        
    Returns:
        Compiled StateGraph for research workflow
    
    Example:
        # With enterprise connectors
        graph = create_research_graph(
            enterprise_connectors=['confluence', 'slack', 'pinecone']
        )
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("planning", create_planning_node(llm))
    workflow.add_node("searching", create_search_node(llm, enterprise_connectors))
    workflow.add_node("reading", create_reading_node(llm))
    workflow.add_node("synthesizing", create_synthesis_node(llm))
    workflow.add_node("error", create_error_node())
    
    # Add edges
    workflow.add_edge("planning", "searching")
    workflow.add_edge("searching", "reading")
    workflow.add_edge("reading", "synthesizing")
    workflow.add_edge("synthesizing", END)
    workflow.add_edge("error", END)
    
    # Set entry point
    workflow.set_entry_point("planning")
    
    # Compile and return
    return workflow.compile()


def run_research(topic: str, llm: Optional[ChatOpenAI] = None) -> str:
    """
    Run the research workflow for a given topic.
    
    Args:
        topic: The research topic
        llm: Optional language model
        
    Returns:
        The final research report
    """
    graph = create_research_graph(llm)
    initial_state = get_initial_state(topic)
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    return final_state["report"]
