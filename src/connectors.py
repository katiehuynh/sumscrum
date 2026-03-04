"""
Custom Enterprise Connectors for the Research Agent.

Add data sources, APIs, and services here.
Each connector follows the same pattern: query in, structured results out.
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from pydantic import BaseModel


# ============================================================
# Connector Base Classes
# ============================================================

class ConnectorResult(BaseModel):
    """Standard result format for all connectors."""
    title: str
    url: str
    content: str
    source_type: str
    metadata: Dict[str, Any] = {}


class BaseConnector:
    """Base class for enterprise connectors."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url
    
    def search(self, query: str, **kwargs) -> List[ConnectorResult]:
        raise NotImplementedError
    
    def get_document(self, doc_id: str) -> ConnectorResult:
        raise NotImplementedError


# ============================================================
# Database Connectors
# ============================================================

@tool
def search_postgresql(
    query: str,
    table: str = "documents",
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search a PostgreSQL database for relevant documents.
    
    Args:
        query: Search query (uses full-text search)
        table: Table to search
        limit: Max results
    
    Returns:
        List of matching documents
    """
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        
        cursor = conn.cursor()
        # Using PostgreSQL full-text search
        cursor.execute(f"""
            SELECT id, title, content, url 
            FROM {table}
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            LIMIT %s
        """, (query, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "title": row[1],
                "url": row[3] or f"db://{table}/{row[0]}",
                "content": row[2][:2000],
                "source_type": "postgresql"
            })
        
        conn.close()
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "postgresql"}]


@tool
def search_elasticsearch(
    query: str,
    index: str = "documents",
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search Elasticsearch for documents.
    
    Args:
        query: Search query
        index: Elasticsearch index
        max_results: Max results
    
    Returns:
        List of matching documents
    """
    try:
        from elasticsearch import Elasticsearch
        
        es = Elasticsearch(
            hosts=[os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")],
            api_key=os.getenv("ELASTICSEARCH_API_KEY")
        )
        
        response = es.search(
            index=index,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "content", "summary"]
                    }
                },
                "size": max_results
            }
        )
        
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append({
                "title": source.get("title", ""),
                "url": source.get("url", f"es://{index}/{hit['_id']}"),
                "content": source.get("content", source.get("summary", ""))[:2000],
                "source_type": "elasticsearch",
                "score": hit["_score"]
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "elasticsearch"}]


# ============================================================
# Cloud Storage Connectors
# ============================================================

@tool
def search_s3_documents(
    query: str,
    bucket: str = None,
    prefix: str = ""
) -> List[Dict[str, Any]]:
    """
    Search documents stored in AWS S3 (requires text extraction).
    
    Args:
        query: Search terms
        bucket: S3 bucket name
        prefix: Key prefix to filter
    
    Returns:
        List of matching documents
    """
    try:
        import boto3
        
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        
        bucket = bucket or os.getenv("S3_BUCKET")
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        results = []
        query_terms = query.lower().split()
        
        for obj in response.get("Contents", [])[:50]:
            key = obj["Key"]
            if key.endswith(('.txt', '.md', '.json')):
                # Get object content
                file_obj = s3.get_object(Bucket=bucket, Key=key)
                content = file_obj["Body"].read().decode("utf-8")[:5000]
                
                # Simple keyword matching
                if any(term in content.lower() for term in query_terms):
                    results.append({
                        "title": key.split("/")[-1],
                        "url": f"s3://{bucket}/{key}",
                        "content": content[:2000],
                        "source_type": "s3"
                    })
        
        return results[:10]
    
    except Exception as e:
        return [{"error": str(e), "source_type": "s3"}]


@tool
def search_azure_blob(
    query: str,
    container: str = None
) -> List[Dict[str, Any]]:
    """
    Search documents in Azure Blob Storage.
    
    Args:
        query: Search terms
        container: Blob container name
    
    Returns:
        List of matching documents
    """
    try:
        from azure.storage.blob import BlobServiceClient
        
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        
        container = container or os.getenv("AZURE_CONTAINER")
        container_client = blob_service.get_container_client(container)
        
        results = []
        query_terms = query.lower().split()
        
        for blob in container_client.list_blobs():
            if blob.name.endswith(('.txt', '.md', '.json')):
                blob_client = container_client.get_blob_client(blob.name)
                content = blob_client.download_blob().readall().decode("utf-8")[:5000]
                
                if any(term in content.lower() for term in query_terms):
                    results.append({
                        "title": blob.name.split("/")[-1],
                        "url": f"azure://{container}/{blob.name}",
                        "content": content[:2000],
                        "source_type": "azure_blob"
                    })
        
        return results[:10]
    
    except Exception as e:
        return [{"error": str(e), "source_type": "azure_blob"}]


# ============================================================
# Enterprise Application Connectors
# ============================================================

@tool
def search_confluence(
    query: str,
    space_key: str = None,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search Atlassian Confluence for documentation.
    
    Args:
        query: Search query
        space_key: Confluence space to search (optional)
        max_results: Max results
    
    Returns:
        List of Confluence pages
    """
    try:
        import requests
        
        base_url = os.getenv("CONFLUENCE_URL")
        auth = (os.getenv("CONFLUENCE_USER"), os.getenv("CONFLUENCE_API_TOKEN"))
        
        params = {
            "cql": f'text ~ "{query}"' + (f' AND space = "{space_key}"' if space_key else ""),
            "limit": max_results,
            "expand": "body.storage"
        }
        
        response = requests.get(
            f"{base_url}/rest/api/content/search",
            params=params,
            auth=auth
        )
        response.raise_for_status()
        
        results = []
        for page in response.json().get("results", []):
            # Strip HTML from content
            from bs4 import BeautifulSoup
            content_html = page.get("body", {}).get("storage", {}).get("value", "")
            content = BeautifulSoup(content_html, "html.parser").get_text()[:2000]
            
            results.append({
                "title": page["title"],
                "url": f"{base_url}/wiki{page['_links']['webui']}",
                "content": content,
                "source_type": "confluence",
                "space": page.get("space", {}).get("key", "")
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "confluence"}]


@tool
def search_sharepoint(
    query: str,
    site: str = None,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search Microsoft SharePoint for documents.
    
    Args:
        query: Search query
        site: SharePoint site (optional)
        max_results: Max results
    
    Returns:
        List of SharePoint documents
    """
    try:
        import requests
        
        # Using Microsoft Graph API
        access_token = os.getenv("SHAREPOINT_ACCESS_TOKEN")
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Search query
        search_url = "https://graph.microsoft.com/v1.0/search/query"
        body = {
            "requests": [{
                "entityTypes": ["driveItem", "listItem"],
                "query": {"queryString": query},
                "from": 0,
                "size": max_results
            }]
        }
        
        response = requests.post(search_url, headers=headers, json=body)
        response.raise_for_status()
        
        results = []
        for hit_container in response.json().get("value", []):
            for hit in hit_container.get("hitsContainers", [{}])[0].get("hits", []):
                resource = hit.get("resource", {})
                results.append({
                    "title": resource.get("name", ""),
                    "url": resource.get("webUrl", ""),
                    "content": hit.get("summary", "")[:2000],
                    "source_type": "sharepoint"
                })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "sharepoint"}]


@tool
def search_notion(
    query: str,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search Notion workspace for pages and databases.
    
    Args:
        query: Search query
        max_results: Max results
    
    Returns:
        List of Notion pages
    """
    try:
        import requests
        
        headers = {
            "Authorization": f"Bearer {os.getenv('NOTION_API_KEY')}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        
        response = requests.post(
            "https://api.notion.com/v1/search",
            headers=headers,
            json={
                "query": query,
                "page_size": max_results
            }
        )
        response.raise_for_status()
        
        results = []
        for page in response.json().get("results", []):
            title = ""
            if page.get("properties", {}).get("title"):
                title_prop = page["properties"]["title"]
                if title_prop.get("title"):
                    title = title_prop["title"][0]["plain_text"] if title_prop["title"] else ""
            elif page.get("properties", {}).get("Name"):
                name_prop = page["properties"]["Name"]
                if name_prop.get("title"):
                    title = name_prop["title"][0]["plain_text"] if name_prop["title"] else ""
            
            results.append({
                "title": title or page.get("id", "Untitled"),
                "url": page.get("url", ""),
                "content": "",  # Would need additional API call to get content
                "source_type": "notion"
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "notion"}]


@tool
def search_slack(
    query: str,
    channel: str = None,
    max_results: int = 20
) -> List[Dict[str, Any]]:
    """
    Search Slack messages for information.
    
    Args:
        query: Search query
        channel: Channel ID to search (optional)
        max_results: Max results
    
    Returns:
        List of Slack messages
    """
    try:
        from slack_sdk import WebClient
        
        client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
        
        response = client.search_messages(
            query=query + (f" in:{channel}" if channel else ""),
            count=max_results
        )
        
        results = []
        for match in response.get("messages", {}).get("matches", []):
            results.append({
                "title": f"Message in #{match.get('channel', {}).get('name', 'unknown')}",
                "url": match.get("permalink", ""),
                "content": match.get("text", "")[:2000],
                "source_type": "slack",
                "user": match.get("user", ""),
                "timestamp": match.get("ts", "")
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "slack"}]


# ============================================================
# API Connectors
# ============================================================

@tool
def search_custom_api(
    query: str,
    endpoint: str = None,
    method: str = "GET"
) -> List[Dict[str, Any]]:
    """
    Search a custom REST API endpoint.
    
    Args:
        query: Search query
        endpoint: API endpoint URL (or use CUSTOM_API_URL env var)
        method: HTTP method
    
    Returns:
        API response formatted as results
    """
    try:
        import requests
        
        endpoint = endpoint or os.getenv("CUSTOM_API_URL")
        api_key = os.getenv("CUSTOM_API_KEY")
        
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        
        if method == "GET":
            response = requests.get(
                endpoint,
                params={"q": query, "query": query, "search": query},
                headers=headers
            )
        else:
            response = requests.post(
                endpoint,
                json={"query": query},
                headers=headers
            )
        
        response.raise_for_status()
        data = response.json()
        
        # Try to normalize response
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("results", data.get("data", data.get("items", [data])))
        else:
            items = [{"content": str(data)}]
        
        results = []
        for item in items[:10]:
            results.append({
                "title": item.get("title", item.get("name", "Result")),
                "url": item.get("url", item.get("link", endpoint)),
                "content": str(item.get("content", item.get("description", item)))[:2000],
                "source_type": "custom_api"
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "custom_api"}]


# ============================================================
# Vector Database Connectors (RAG)
# ============================================================

@tool
def search_pinecone(
    query: str,
    namespace: str = "",
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search Pinecone vector database for similar documents.
    
    Args:
        query: Search query (will be embedded)
        namespace: Pinecone namespace
        top_k: Number of results
    
    Returns:
        List of similar documents
    """
    try:
        from pinecone import Pinecone
        from langchain_openai import OpenAIEmbeddings
        
        # Initialize
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX"))
        embeddings = OpenAIEmbeddings()
        
        # Embed query
        query_embedding = embeddings.embed_query(query)
        
        # Search
        response = index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        
        results = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            results.append({
                "title": metadata.get("title", match["id"]),
                "url": metadata.get("url", f"pinecone://{match['id']}"),
                "content": metadata.get("text", metadata.get("content", ""))[:2000],
                "source_type": "pinecone",
                "score": match.get("score", 0)
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "pinecone"}]


@tool
def search_chromadb(
    query: str,
    collection: str = "documents",
    n_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search ChromaDB for similar documents.
    
    Args:
        query: Search query
        collection: Collection name
        n_results: Number of results
    
    Returns:
        List of similar documents
    """
    try:
        import chromadb
        
        client = chromadb.PersistentClient(
            path=os.getenv("CHROMADB_PATH", "./chromadb")
        )
        collection = client.get_collection(collection)
        
        results_data = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        results = []
        for i, doc_id in enumerate(results_data["ids"][0]):
            results.append({
                "title": results_data["metadatas"][0][i].get("title", doc_id),
                "url": results_data["metadatas"][0][i].get("url", f"chroma://{doc_id}"),
                "content": results_data["documents"][0][i][:2000],
                "source_type": "chromadb"
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e), "source_type": "chromadb"}]


# ============================================================
# Connector Registry
# ============================================================

def get_all_enterprise_connectors():
    """Get all available enterprise connectors."""
    return [
        # Databases
        search_postgresql,
        search_elasticsearch,
        # Cloud Storage
        search_s3_documents,
        search_azure_blob,
        # Enterprise Apps
        search_confluence,
        search_sharepoint,
        search_notion,
        search_slack,
        # APIs
        search_custom_api,
        # Vector DBs
        search_pinecone,
        search_chromadb,
    ]


def get_connector_by_name(name: str):
    """Get a specific connector by name."""
    connectors = {
        "postgresql": search_postgresql,
        "elasticsearch": search_elasticsearch,
        "s3": search_s3_documents,
        "azure": search_azure_blob,
        "confluence": search_confluence,
        "sharepoint": search_sharepoint,
        "notion": search_notion,
        "slack": search_slack,
        "custom_api": search_custom_api,
        "pinecone": search_pinecone,
        "chromadb": search_chromadb,
    }
    return connectors.get(name)
