import asyncio
from typing import List, Optional

from dto import AgentCard, AgentCardCapability, AgentCardEndpoint
from utils import getValueFromConfig


class _A2AManager:
    """Internal A2A manager (not exposed directly)"""

    def __init__(self):
        self._agent_card = None
        self._initialized = False
        self._lock = asyncio.Lock()


    def _load_a2a_config(self) -> dict:
        """Load A2A configuration from config.toml"""
        config = getValueFromConfig("a2a", "card_details")
        return config.get("a2a", {
            "enabled": True,
            "agent_id": "code-mentor-agent",
            "agent_name": "Code Mentor RAG Agent",
            "description": "Analyzes git commits and provides code insights using semantic search",
            "base_url": "http://localhost:8001"
        })

    def _generate_agent_card(
        self,
        agent_id: str,
        agent_name: str,
        description: str,
        base_url: str,
        capabilities: Optional[List[AgentCardCapability]] = None,
        supports_streaming: bool = False
    ) -> AgentCard:
        if capabilities is None:
            capabilities = []
        
        endpoint = AgentCardEndpoint(
            url=f"{base_url}/a2a/{agent_id}",
            method="POST",
            protocol="a2a/v1"
        )
        
        return AgentCard(
            id=agent_id,
            name=agent_name,
            description=description,
            endpoint=endpoint,
            capabilities=capabilities,
            supports_streaming=supports_streaming,
            supports_context=True,
            input_modes=["text"],
            output_modes=["text"]
        )
    
    async def initialize(self):
        """Initialize the A2A components"""
        async with self._lock:
            if not self._initialized:
                print("Initializing A2A...")
                config = self._load_a2a_config()
        
                # Define capabilities
                capabilities = [
                    AgentCardCapability(
                        name="code_analysis",
                        description="Search through git commits and diffs using semantic search. Analyze the code changes and provide insights",
                        parameters={
                            "query": {"type": "string", "description": "Analysis query"}
                        }
                    )
                ]
                
                self._agent_card = self._generate_agent_card(
                    agent_id=config["agent_id"],
                    agent_name=config["agent_name"],
                    description=config["description"],
                    base_url=config["base_url"],
                    capabilities=capabilities,
                    supports_streaming=True
                )
                
                self._initialized = True
                print("A2A initialized successfully!")

    @property
    def agent_card(self):
        if not self._initialized:
            raise RuntimeError("A2A not initialized. Call 'await initialize()' first.")
        return self._agent_card

    def is_initialized(self) -> bool:
        return self._initialized

# Module-level singleton instance
_manager = _A2AManager()

async def initialize_a2a():
    """Initialize the A2A components"""
    await _manager.initialize()

def get_agent_card() -> AgentCard:
    """Get or create the agent card"""
    return _manager.agent_card

def is_a2a_initialized() -> bool:
    """Check if A2A is initialized"""
    return _manager.is_initialized()