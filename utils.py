from typing import Any, List
from langchain.messages import AnyMessage, HumanMessage, ToolMessage
import tomllib

from dto import MessageDetail

def convert_arg_types(value):
    """Convert string representations to proper types"""
    if not isinstance(value, str):
        return value
    
    # Handle booleans
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    
    # Handle integers
    try:
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
    except (ValueError, AttributeError):
        pass
    
    # Handle floats
    try:
        if '.' in value:
            return float(value)
    except ValueError:
        pass
    
    # Return as-is if no conversion needed
    return value

def getValueFromConfig(root_key, key):
    config_path = "config.toml"
    config = {}
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    tools_config = config.get(root_key, {})
    return tools_config[key] if key in tools_config else None


class MessageConverter:
    """Utilities for converting between internal LangChain format and A2A format"""
    
    @staticmethod
    def langchain_to_raw(langchain_messages: List[AnyMessage]) -> List[MessageDetail]:
        a2a_messages = []
        
        for msg in langchain_messages:
            # Handle different LangChain message types
            msg_type = msg.__class__.__name__
            
            if msg_type == "HumanMessage":
                a2a_messages.append(MessageDetail(
                    role="user",
                    content=str(msg.content)
                ))
            elif msg_type == "AIMessage":
                # Handle tool calls if present
                tool_calls = None
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls = msg.tool_calls
                
                a2a_messages.append(MessageDetail(
                    role="assistant",
                    content=str(msg.content),
                    tool_calls=tool_calls
                ))
            elif msg_type == "SystemMessage":
                a2a_messages.append(MessageDetail(
                    role="system",
                    content=str(msg.content)
                ))
            elif msg_type == "ToolMessage":
                a2a_messages.append(MessageDetail(
                    role="tool",
                    content=str(msg.content),
                    tool_call_id=getattr(msg, 'tool_call_id', None)
                ))
        
        return a2a_messages
    
    @staticmethod
    def raw_to_langchain(a2a_messages: List[MessageDetail]) -> List[Any]:
        from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        
        langchain_messages = []
        
        for msg in a2a_messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                ai_msg = AIMessage(content=msg.content)
                if msg.tool_calls:
                    ai_msg.tool_calls = msg.tool_calls
                langchain_messages.append(ai_msg)
            elif msg.role == "system":
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "tool":
                langchain_messages.append(ToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_call_id or ""
                ))
        
        return langchain_messages