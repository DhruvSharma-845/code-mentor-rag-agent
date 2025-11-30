import json
from langchain.messages import HumanMessage

from agent_manager import get_agent, get_checkpointer
from dto import ConversationHistory, MessageDetail
from utils import MessageConverter

async def get_state(config: dict) -> dict:
    return await get_agent().aget_state(config)

async def chat_with_agent(thread_id: str, input_messages: list[MessageDetail], user_id: str) -> ConversationHistory:
    # Use thread_id from request for conversation tracking
    langchain_messages = MessageConverter.raw_to_langchain(input_messages)
    config = {"configurable": {"thread_id": f"{user_id}_{thread_id}", "user_id": user_id}}
    result = await get_agent().ainvoke({"messages": langchain_messages}, config=config)
    # Get the state from the checkpointer
    state = await get_state(config)
    
    if not state or not state.values.get("messages"):
        return ConversationHistory(thread_id=thread_id, messages=[], user_id=user_id)

    filtered_messages = []
    for msg in state.values["messages"][::-1]:
        if isinstance(msg, HumanMessage):
            break
        filtered_messages.append(msg)

    messages = MessageConverter.langchain_to_raw(filtered_messages[::-1])
        
    return ConversationHistory(thread_id=thread_id, messages=messages, user_id=user_id)

async def chat_with_agent_stream_generator(thread_id: str, input_messages: list[MessageDetail], user_id: str):
    langchain_messages = MessageConverter.raw_to_langchain(input_messages)
    config = {"configurable": {"thread_id": f"{user_id}_{thread_id}", "user_id": user_id}}
    
    async for chunk in get_agent().astream({"messages": langchain_messages}, config, stream_mode="updates"):
        messages = []
        if "llm_call" in chunk:
            messages = chunk["llm_call"]["messages"]
        elif "tool_node" in chunk:
            messages = chunk["tool_node"]["messages"]
        
        if len(messages) > 0:
            messages = MessageConverter.langchain_to_raw(messages)
            conv_history = ConversationHistory(thread_id=thread_id, messages=messages, user_id=user_id)
            yield conv_history
    
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
    messages = MessageConverter.langchain_to_raw(state.values["messages"])
    
    return ConversationHistory(thread_id=thread_id, messages=messages, user_id=user_id)
    
        