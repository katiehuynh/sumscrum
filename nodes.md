## **Flow When Connectors Are Configured**

```
create_search_node(llm, enterprise_connectors=['confluence', 'slack'])
                                    │
                                    ▼
                         search_node(state) is called
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
         Enterprise Connectors              Web + ArXiv Search
         (runs FIRST if configured)         (always runs)
                    │                               │
                    ▼                               ▼
    ┌──────────────────────────────┐   ┌─────────────────────────┐
    │ for connector_name in        │   │ search_web()            │
    │   ['confluence', 'slack']:   │   │ search_arxiv()          │
    │                              │   │                         │
    │ 1. get_connector_by_name()   │   │ (public web sources)    │
    │ 2. connector.invoke(topic)   │   │                         │
    │ 3. Add results to all_sources│   │                         │
    └──────────────────────────────┘   └─────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                          all_sources combined
                          (internal + web + papers)
```

## **Step-by-Step Execution**

```python
# When you create the agent with connectors:
agent = ResearchAgent(enterprise_connectors=['confluence', 'slack'])

# During the SEARCH phase, this happens:

# STEP 1: Get connector by name
connector = get_connector_by_name('confluence')  
# Returns: search_confluence function from connectors.py

# STEP 2: Call the connector with the research topic
results = connector.invoke({"query": "Quantum Computing"})
# This calls search_confluence() which:
#   - Connects to your Confluence API
#   - Searches for "Quantum Computing" in your wiki
#   - Returns internal documentation

# STEP 3: Add results to source pool
all_sources.append({
    "title": "Internal: Quantum Computing Overview",
    "url": "https://yourcompany.atlassian.net/wiki/...",
    "content": "Our team's notes on quantum computing...",
    "source_type": "confluence",
    "subtopic": "internal knowledge"  # Tagged as internal
})

# STEP 4: Repeat for 'slack'
connector = get_connector_by_name('slack')
results = connector.invoke({"query": "Quantum Computing"})
# Searches Slack messages mentioning the topic
```

## **What This Enables**

| Without Connectors | With Connectors |
|-------------------|-----------------|
| Only searches public web | Also searches your internal data |
| Generic information | Company-specific knowledge included |
| No proprietary data | Internal docs, discussions, wikis |

## **Example Output Difference**

**Without connectors** — Report only cites:
```
[1] Wikipedia - "Quantum Computing"
[2] ArXiv paper - "Introduction to Qubits"
[3] MIT Tech Review article
```

**With connectors (confluence + slack)** — Report also includes:
```
[1] Wikipedia - "Quantum Computing"
[2] ArXiv paper - "Introduction to Qubits"  
[3] MIT Tech Review article
[4] Internal Wiki - "Our Quantum Computing Strategy"     ← From Confluence
[5] Slack discussion - "#research: quantum computing"     ← From Slack
```

## **Configuration Required**

For connectors to work, you need the API credentials in .env:

```env
# For Confluence
CONFLUENCE_URL=https://yourcompany.atlassian.net
CONFLUENCE_USER=your-email@company.com
CONFLUENCE_API_TOKEN=your_api_token

# For Slack
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
```

The connectors are **optional** — if not configured or credentials missing, the agent gracefully falls back to just web search.