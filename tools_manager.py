# Semantic search tools
import asyncio
import tomllib
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from tools import getTools
import uuid

from utils import getValueFromConfig


class _ToolsManager:
    """Internal tools manager (not exposed directly)"""
    
    def __init__(self):
        # self._vector_store = None
        self._tools = None
        # self._tool_registry = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the tools components"""
        async with self._lock:
            if not self._initialized:
                print("Initializing Tools...")

                # semantic_search_embeddings_model = getValueFromConfig("tools", "semantic_search_embeddings_model")

                # embeddings = OllamaEmbeddings(model=semantic_search_embeddings_model)
                # self._vector_store = InMemoryVectorStore(embeddings)
                
                self._tools = await getTools()
                # self._tool_registry = {str(uuid.uuid4()): tool for tool in self._tools}

                # tool_documents = [
                #     Document(
                #         page_content=tool.description,
                #         id=id,
                #         metadata={"tool_name": tool.name},
                #     )
                #     for id, tool in self._tool_registry.items()
                # ]
                # self._vector_store.add_documents(tool_documents)

                self._initialized = True
                print("Tools initialized successfully!")

    async def cleanup(self):
        """Cleanup the tools components"""
        async with self._lock:
            if self._initialized:
                print("Cleaning up Tools...")

                # self._vector_store = None
                self._tools = None
                # self._tool_registry = None
                self._initialized = False
                print("Tools cleaned up successfully!")

    # @property
    # def vector_store(self):
    #     """Get the vector store instance"""
    #     return self._vector_store

    @property
    def tools(self):
        """Get the tools instance"""
        return self._tools

    # @property
    # def tool_registry(self):
    #     """Get the tool registry instance"""
    #     return self._tool_registry

_manager = _ToolsManager()

async def initialize_tools():
    """Initialize the tools components"""
    await _manager.initialize()

async def cleanup_tools():
    """Cleanup the tools components"""
    await _manager.cleanup()

# def get_vector_store():
#     """Get the vector store instance"""
#     return _manager.vector_store

# def get_tool_registry():
#     """Get the tool registry instance"""
#     return _manager.tool_registry

def get_tools():
    """Get the tools instance"""
    return _manager.tools

# def get_tools_by_query(query: str):
#     """Get the tools by query"""
#     semantic_search_enabled = getValueFromConfig("tools", "semantic_search_enabled")
#     if semantic_search_enabled == "true":
#         # vector store would have been created till now
#         # in the app boostrap
#         documents =_manager.vector_store.similarity_search(query)
#         return [document.id for document in documents]
#     else:
#         return [id for id in get_tool_registry().keys()]