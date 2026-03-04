# Design Patterns & Agent Orchestration Analysis

## 1. State Machine Pattern (Finite State Machine)

The core of this agent is a **State Machine** implemented via LangGraph.

```
┌─────────────────────────────────────────────────────────────────┐
│                         ResearchState                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ topic: str           │ The research question             │   │
│  │ subtopics: List[str] │ Decomposed sub-questions          │   │
│  │ sources: List[Dict]  │ Collected source materials        │   │
│  │ findings: List[str]  │ Extracted insights                │   │
│  │ report: str          │ Final output                      │   │
│  │ current_step: str    │ Which state we're in              │   │
│  │ messages: List       │ Conversation history              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Why this pattern?**
- State is **explicit and inspectable** at any point
- Each transition is **deterministic** (planning → searching → reading → synthesizing)
- Easy to **debug** — you can see exactly what state caused an issue
- Supports **checkpointing** — can save/resume from any state

**Code location:** graph.py

---

## 2. Pipeline Pattern (Sequential Workflow)

The agent uses a **linear pipeline** where each stage transforms the state:

```
INPUT                                                         OUTPUT
  │                                                              │
  ▼                                                              ▼
┌──────────┐    ┌───────────┐    ┌─────────┐    ┌─────────────┐
│ Planning │───▶│ Searching │───▶│ Reading │───▶│ Synthesizing│───▶ Report
└──────────┘    └───────────┘    └─────────┘    └─────────────┘
     │               │               │                │
     ▼               ▼               ▼                ▼
  subtopics       sources        findings          report
```

**Each stage has a single responsibility:**

| Stage | Input | Output | Responsibility |
|-------|-------|--------|----------------|
| Planning | topic | subtopics | Decompose question |
| Searching | subtopics | sources | Find information |
| Reading | sources | findings | Extract insights |
| Synthesizing | findings | report | Generate output |

**Code location:** graph.py

---

## 3. Factory Pattern

The agent uses **factory functions** to create nodes:

```python
def create_planning_node(llm: ChatOpenAI):    # Factory
    def planning_node(state: ResearchState):   # Product
        # ... implementation
        return updated_state
    return planning_node                        # Returns the product
```

**Why factories?**
- **Dependency injection**: Each node gets its own LLM instance
- **Encapsulation**: Node logic is hidden inside the factory
- **Testability**: Easy to mock the LLM for testing
- **Configurability**: Can create nodes with different parameters

**Code locations:**
- `create_planning_node()` — graph.py
- `create_search_node()` — graph.py
- `create_reading_node()` — graph.py
- `create_synthesis_node()` — graph.py

---

## 4. Facade Pattern

The `ResearchAgent` class is a **Facade** that hides complexity:

```python
# Without Facade - user must understand internals:
llm = ChatOpenAI(model="gpt-4o-mini")
graph = StateGraph(ResearchState)
graph.add_node("planning", create_planning_node(llm))
# ... lots of setup
state = get_initial_state(topic)
result = graph.invoke(state)

# With Facade - simple interface:
agent = ResearchAgent()
report = agent.research("Quantum Computing")
```

**Benefits:**
- Users don't need to understand LangGraph, state machines, or node configuration
- Single entry point for all functionality
- Internal implementation can change without affecting users

**Code location:** agent.py

---

## 5. Strategy Pattern

Tools are interchangeable **strategies** for accomplishing tasks:

```python
# Different search strategies (same interface)
search_web()      # Strategy 1: Web search via Tavily
search_arxiv()    # Strategy 2: Academic papers via ArXiv

# Different reading strategies
read_webpage()    # Strategy 1: Web content
# Could add: read_pdf(), read_youtube_transcript(), etc.
```

**Why this matters:**
- Can swap `search_web` for a different provider without changing the agent
- Easy to add new tools (PDF reader, YouTube transcripts, etc.)
- The graph doesn't care which tool is used — same interface

**Code location:** tools.py

---

## 6. Observer Pattern (Callback System)

The agent supports **callbacks** for monitoring progress:

```python
def my_callback(step_name: str, state: dict):
    print(f"Completed: {step_name}")
    save_checkpoint(state)  # Custom behavior

report = agent.research("Topic", callback=my_callback)
```

**In the code:**
```python
for step_output in self.graph.stream(initial_state):
    step_name = list(step_output.keys())[0]
    if callback:
        callback(step_name, step_state)  # Notify observer
```

**Use cases:**
- Progress bars
- Logging
- Checkpointing
- Analytics

**Code location:** agent.py

---

## 7. Agent Orchestration Patterns

### 7.1 DAG-Based Orchestration (Directed Acyclic Graph)

LangGraph implements **DAG orchestration** — a graph of nodes where:
- Each node is an **agent capability** (plan, search, read, synthesize)
- Edges define **execution order**
- State flows through the graph

```python
workflow = StateGraph(ResearchState)
workflow.add_node("planning", ...)
workflow.add_node("searching", ...)
workflow.add_edge("planning", "searching")  # DAG edge
```

### 7.2 Shared State Orchestration

All nodes share and mutate a **single state object**:

```python
def planning_node(state: ResearchState) -> ResearchState:
    # Read from state
    topic = state["topic"]
    
    # ... do work ...
    
    # Return modified state
    return {
        **state,                    # Keep existing state
        "subtopics": subtopics,     # Add new data
        "current_step": "searching" # Update step
    }
```

**This is different from:**
- **Message passing**: Nodes send messages to each other
- **Blackboard**: Shared memory without structure
- **Event-driven**: Nodes react to events

### 7.3 Tool-Augmented LLM Pattern

Each node combines an **LLM with tools**:

```
┌─────────────────────────────────────────┐
│              Search Node                │
│  ┌─────────┐        ┌───────────────┐  │
│  │   LLM   │───────▶│  Tools        │  │
│  │ (GPT-4) │        │ - search_web  │  │
│  └─────────┘        │ - search_arxiv│  │
│       │             └───────────────┘  │
│       ▼                    │           │
│   Reasoning            Actions         │
└─────────────────────────────────────────┘
```

**The LLM provides:**
- Reasoning (what to search for)
- Synthesis (extracting key points)
- Generation (writing the report)

**Tools provide:**
- Actions (actually searching the web)
- Data retrieval (fetching paper abstracts)

---

## 8. Architectural Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RESEARCH AGENT SYSTEM                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      FACADE LAYER (agent.py)                     │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐   │    │
│  │  │ research()  │  │ quick_search │  │ search_papers()       │   │    │
│  │  └─────────────┘  └──────────────┘  └───────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                 ORCHESTRATION LAYER (graph.py)                   │    │
│  │                                                                  │    │
│  │   ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌────────────┐   │    │
│  │   │ Planning │──▶│ Searching│──▶│ Reading │──▶│Synthesizing│   │    │
│  │   │   Node   │   │   Node   │   │  Node   │   │    Node    │   │    │
│  │   └──────────┘   └──────────┘   └─────────┘   └────────────┘   │    │
│  │        │              │              │              │          │    │
│  │        └──────────────┴──────────────┴──────────────┘          │    │
│  │                              │                                  │    │
│  │                    ┌─────────▼─────────┐                       │    │
│  │                    │   ResearchState   │                       │    │
│  │                    │  (Shared State)   │                       │    │
│  │                    └───────────────────┘                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      TOOLS LAYER (tools.py)                      │    │
│  │  ┌────────────┐  ┌─────────────┐  ┌────────────┐                │    │
│  │  │ search_web │  │search_arxiv │  │read_webpage│                │    │
│  │  └────────────┘  └─────────────┘  └────────────┘                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    EXTERNAL SERVICES                             │    │
│  │  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────┐  │    │
│  │  │  OpenAI  │  │  Tavily   │  │   ArXiv    │  │  Web Pages   │  │    │
│  │  │   API    │  │    API    │  │    API     │  │   (HTTP)     │  │    │
│  │  └──────────┘  └───────────┘  └────────────┘  └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Comparison with Other Orchestration Styles

| Approach | This Agent | ReAct Agents | Multi-Agent |
|----------|------------|--------------|-------------|
| **Structure** | Fixed DAG | Loop until done | Agents communicate |
| **Control Flow** | Deterministic | LLM decides | Negotiation |
| **Predictability** | High | Low | Medium |
| **Debuggability** | Easy | Hard | Medium |
| **Flexibility** | Medium | High | High |
| **Cost** | Predictable | Variable | Variable |

---

## 10. Key Takeaways

1. **State Machine** makes execution predictable and debuggable
2. **Pipeline Pattern** ensures each stage has one job
3. **Factory Pattern** enables dependency injection for testing
4. **Facade Pattern** hides complexity from users
5. **Strategy Pattern** makes tools interchangeable
6. **Shared State** allows nodes to build on each other's work
7. **Tool-Augmented LLM** combines reasoning with actions

This architecture is ideal for **predictable, multi-step workflows** where you want reliability over flexibility. For more dynamic tasks, you might add conditional edges or loops.