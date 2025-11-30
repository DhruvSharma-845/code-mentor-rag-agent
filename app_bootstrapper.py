from a2a_manager import initialize_a2a
from agent_manager import cleanup_agent, initialize_agent
from knowledge_base_handler import cleanup_knowledge_base, initialize_knowledge_base
from tools_manager import cleanup_tools, initialize_tools
# from memory_store import cleanup_memory_store, initialize_memory_store
# from tools_manager import cleanup_tools, initialize_tools


async def bootstrap_app():
    # await initialize_memory_store()
    await initialize_tools()
    await initialize_knowledge_base()
    await initialize_agent()
    await initialize_a2a()

async def destroy_app():
    await cleanup_agent()
    await cleanup_tools()
    # await cleanup_memory_store()
    await cleanup_knowledge_base()