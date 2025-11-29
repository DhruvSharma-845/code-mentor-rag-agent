from langchain.tools import tool

from knowledge_base_handler import get_knowledge_store


@tool(response_format="content_and_artifact")
def search_git_commits_based_on_query(query: str, limit: int = 10):
    """
    Search through git commits and diffs using semantic search.
    
    Args:
        query: The search query (e.g., "authentication changes", "bug fixes in login")
        limit: Maximum number of results to return
    
    Returns:
        List of relevant documents with their similarity scores
    """
    store = get_knowledge_store()
    if not store:
        return []
    
    # Search with similarity scores
    results = store.similarity_search_with_score(
        query=query,
        k=limit,
        filter={"source": "git_commit"}  # Only search git commits
    )

    serialized = "\n\n".join(
        (f"Metadata: {doc.metadata}\nContent: {doc.page_content}")
        for doc, score in results
    )

    retrieved_docs = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        for doc, score in results
    ]
    
    return serialized, retrieved_docs


async def getTools():
    # mcpServersConfig = getMCPServersConfig()
    # client = MultiServerMCPClient(mcpServersConfig)
    # tools = await client.get_tools()

    tools = []

    tools.extend([search_git_commits_based_on_query])

    # tools.append(getWikipediaTool())

    return tools