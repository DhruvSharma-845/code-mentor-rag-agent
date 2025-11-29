from langchain.messages import HumanMessage, ToolMessage
import tomllib

def get_messages_details(conv_messages, thread_id):
    # Extract and format messages
    messages = []
    for msg in conv_messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        else:
            role = "assistant"
        
        content = msg.content
        if getattr(msg, "tool_calls", None) is not None:
            for tool_call in msg.tool_calls:
                content += f"Tool call: {tool_call['name']} with args: {tool_call['args']}"

        messages.append({
            "id": thread_id,
            "role": role,
            "content": content,
        })
    return messages

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