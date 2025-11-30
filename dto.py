from datetime import datetime
from typing import Any, List, Literal, Optional, Dict
import uuid
from pydantic import BaseModel, Field

class MessageDetail(BaseModel):
    id: Optional[str] = None
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    messages: list[MessageDetail]
    thread_id: str
    user_id: str

class ConversationHistory(BaseModel):
    thread_id: str
    messages: list[MessageDetail]
    user_id: str

class A2ARequest(BaseModel):
    """
    A2A protocol request format (JSON-RPC style).
    Sent to /a2a/{assistant_id} endpoint.
    """
    jsonrpc: Literal["2.0"] = "2.0"
    method: Literal["invoke", "stream"] = "invoke"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Request parameters
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request parameters including messages and config"
    )
    
    # Core fields within params
    def __init__(self, **data):
        super().__init__(**data)
        if "params" not in data or not data["params"]:
            self.params = {
                "messages": [],
                "config": {},
                "thread_id": None,
                "user_id": None
            }
    
    @property
    def messages(self) -> List[MessageDetail]:
        """Get messages from params"""
        msgs = self.params.get("messages", [])
        return [MessageDetail(**m) if isinstance(m, dict) else m for m in msgs]
    
    @property
    def thread_id(self) -> Optional[str]:
        """Get thread_id from params"""
        return self.params.get("thread_id")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get config from params"""
        return self.params.get("config", {})

    @property
    def user_id(self) -> Optional[str]:
        """Get user_id from params"""
        return self.params.get("user_id")

class A2AResponse(BaseModel):
    """A2A protocol response format (JSON-RPC style)"""
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    
    # Either result or error
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    
    @classmethod
    def success(cls, request_id: str, messages: List[MessageDetail], thread_id: str, user_id: str, metadata: Optional[Dict] = None):
        """Create a successful response"""
        return cls(
            id=request_id,
            result={
                "messages": [msg.model_dump() for msg in messages],
                "thread_id": thread_id,
                "user_id": user_id,
                "metadata": metadata or {}
            }
        )
    
    @classmethod
    def error_response(cls, request_id: str, code: int, message: str, data: Optional[Any] = None):
        """Create an error response"""
        error_obj = {
            "code": code,
            "message": message
        }
        if data:
            error_obj["data"] = data
        return cls(id=request_id, error=error_obj)

class A2AStreamCompletionResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: Optional[Dict[str, Any]] = None


class AgentCardCapability(BaseModel):
    """Describes a capability that the agent supports"""
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None

class AgentCardEndpoint(BaseModel):
    """A2A endpoint information"""
    url: str
    method: Literal["POST"] = "POST"
    protocol: Literal["a2a/v1"] = "a2a/v1"
    
class AgentCard(BaseModel):
    """
    Agent Card following the A2A protocol specification.
    Served at /.well-known/agent-card.json
    """
    # Basic information
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Description of agent capabilities")
    version: str = Field(default="1.0.0", description="Agent version")
    
    # Capabilities
    capabilities: List[AgentCardCapability] = Field(
        default_factory=list,
        description="List of capabilities this agent supports"
    )
    
    # Communication details
    endpoint: AgentCardEndpoint = Field(..., description="A2A endpoint information")
    
    # Supported modes
    input_modes: List[str] = Field(
        default=["text"],
        description="Supported input modes (text, image, audio, etc.)"
    )
    output_modes: List[str] = Field(
        default=["text"],
        description="Supported output modes"
    )
    
    # Additional metadata
    supports_streaming: bool = Field(
        default=False,
        description="Whether the agent supports streaming responses"
    )
    supports_context: bool = Field(
        default=True,
        description="Whether the agent maintains conversation context"
    )
    
    # Optional fields
    documentation_url: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class A2AErrorCode:
    """Standard A2A error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom error codes
    AGENT_NOT_FOUND = -32001
    AGENT_BUSY = -32002
    RATE_LIMITED = -32003
    CONTEXT_TOO_LONG = -32004
