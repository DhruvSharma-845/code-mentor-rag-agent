import asyncio
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# from memory_store import get_memory_store
from model import getModel
from agent import getAgent
from tools import getTools


class _AgentManager:
    """Internal agent manager (not exposed directly)"""
    
    def __init__(self):
        self._agent = None
        self._model = None
        self._tools = None
        self._initialized = False
        self._checkpointer = None 
        self._checkpointer_context = None
        # self._memory_store = None
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the agent components"""
        async with self._lock:
            if not self._initialized:
                print("Initializing Agent...")

                # Use SQLite instead of PostgreSQL - much simpler!
                self._checkpointer_context = AsyncSqliteSaver.from_conn_string("checkpoints.db")
                self._checkpointer = await self._checkpointer_context.__aenter__()
                
                self._model = getModel()

                # get all available tools
                self._tools = await getTools()
                # self._memory_store = get_memory_store()
                self._agent = getAgent(self._model, self._checkpointer, self._tools)

                
                self._initialized = True
                print("Agent initialized successfully!")
    
    async def cleanup(self):
        """Cleanup resources"""
        async with self._lock:
            if self._initialized and self._checkpointer_context:
                print("Cleaning up agent resources...")
                try:
                    await self._checkpointer_context.__aexit__(None, None, None)
                except Exception as e:
                    print(f"Error during cleanup: {e}")
                finally:
                    self._checkpointer = None
                    self._checkpointer_context = None
                    self._agent = None
                    self._model = None
                    self._tools = None
                    self._initialized = False
                    # self._memory_store = None
                    print("Cleanup complete!")

    @property
    def agent(self):
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call 'await initialize_agent()' first.")
        return self._agent
    
    @property
    def model(self):
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call 'await initialize_agent()' first.")
        return self._model
    
    @property
    def tools(self):
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call 'await initialize_agent()' first.")
        return self._tools

    @property
    def checkpointer(self):
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call 'await initialize_agent()' first.")
        return self._checkpointer
    
    def is_initialized(self) -> bool:
        return self._initialized


# Module-level singleton instance
_manager = _AgentManager()


# Public API
async def initialize_agent():
    """Initialize the agent singleton"""
    await _manager.initialize()

async def cleanup_agent():
    """Cleanup the agent singleton"""
    await _manager.cleanup()

def get_agent():
    """Get the agent instance"""
    return _manager.agent


def get_model():
    """Get the model instance"""
    return _manager.model


def get_tools():
    """Get the tools instance"""
    return _manager.tools

def get_checkpointer():
    """Get the checkpointer instance"""
    return _manager.checkpointer

def is_agent_initialized() -> bool:
    """Check if agent is initialized"""
    return _manager.is_initialized()