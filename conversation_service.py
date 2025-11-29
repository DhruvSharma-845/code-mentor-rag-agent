import json
from langchain.messages import HumanMessage

from agent_manager import get_agent, get_checkpointer
from dto import ConversationHistory
from utils import get_messages_details

async def get_state(config: dict) -> dict:
    return await get_agent().aget_state(config)

async def chat_with_agent(thread_id: str, input_messages: list[str], user_id: str) -> ConversationHistory:
    conversation_messages = [HumanMessage(content=message) for message in input_messages]
    # Use thread_id from request for conversation tracking
    config = {"configurable": {"thread_id": f"{user_id}_{thread_id}", "user_id": user_id}}
    result = await get_agent().ainvoke({"messages": conversation_messages}, config=config)
    # Get the state from the checkpointer
    state = await get_state(config)
    
    if not state or not state.values.get("messages"):
        return ConversationHistory(thread_id=thread_id, messages=[], user_id=user_id)

    filtered_messages = []
    for msg in state.values["messages"][::-1]:
        if isinstance(msg, HumanMessage):
            break
        filtered_messages.append(msg)

    messages = get_messages_details(filtered_messages[::-1], thread_id)
        
    return ConversationHistory(thread_id=thread_id, messages=messages, user_id=user_id)

async def chat_with_agent_stream_generator(thread_id: str, input_messages: list[str], user_id: str):
    conversation_messages = [HumanMessage(content=message) for message in input_messages]
    config = {"configurable": {"thread_id": f"{user_id}_{thread_id}", "user_id": user_id}}
    
    try:
        async for chunk in get_agent().astream({"messages": conversation_messages}, config, stream_mode="updates"):
            messages = []
            if "llm_call" in chunk:
                messages = chunk["llm_call"]["messages"]
            elif "tool_node" in chunk:
                messages = chunk["tool_node"]["messages"]
            
            if len(messages) > 0:
                messages = get_messages_details(messages, thread_id)
                conv_history = ConversationHistory(thread_id=thread_id, messages=messages, user_id=user_id)
                yield f"data: {conv_history.model_dump_json()}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
async def get_all_conversation_ids(user_id: str) ->list[str]:
    thread_ids = set()
    prefix = f"{user_id}_"
    # List all checkpoints and extract unique thread_ids
    async for checkpoint_tuple in get_checkpointer().alist(None):
        config = checkpoint_tuple.config
        if config and "configurable" in config and "thread_id" in config["configurable"]:
            full_thread_id = config["configurable"]["thread_id"]
            # Only include threads that belong to this user
            if full_thread_id.startswith(prefix):
                # Strip the user_id prefix to return just the thread_id
                thread_ids.add(full_thread_id[len(prefix):])

    
    return sorted(list(thread_ids))

async def get_conversation_history_from_agent(thread_id: str, user_id: str) -> ConversationHistory:
    config = {"configurable": {"thread_id": f"{user_id}_{thread_id}", "user_id": user_id}}
    
    # Get the state from the checkpointer
    state = await get_state(config)
    
    if not state or not state.values.get("messages"):
        return ConversationHistory(thread_id=thread_id, messages=[], user_id=user_id)
    
    # Extract and format messages
    messages = get_messages_details(state.values["messages"], thread_id)
    
    return ConversationHistory(thread_id=thread_id, messages=messages, user_id=user_id)
    
        