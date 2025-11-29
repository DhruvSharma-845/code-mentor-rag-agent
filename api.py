from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 

from contextlib import asynccontextmanager

from fastapi.responses import StreamingResponse

from app_bootstrapper import bootstrap_app, destroy_app
from conversation_service import chat_with_agent, chat_with_agent_stream_generator, get_all_conversation_ids, get_conversation_history_from_agent
from dto import ChatRequest, ConversationHistory

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up...")
    await bootstrap_app()
    
    yield
    print("Shutting down...")
    await destroy_app()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"Hello": "World"}

@app.post("/api/universal-agent/chat")
async def universal_agent_chat(request: ChatRequest) -> ConversationHistory:
    return await chat_with_agent(request.thread_id, request.messages, request.user_id)

@app.post("/api/universal-agent/chat/stream")
async def universal_agent_chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chat_with_agent_stream_generator(request.thread_id, request.messages, request.user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            }
        )

@app.get("/api/conversations")
async def list_all_conversations(user_id: str) -> dict:
    # try:
        thread_ids = await get_all_conversation_ids(user_id)
        return {"threads": thread_ids, "count": len(thread_ids)}
    # except Exception as e:
        # raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@app.get("/api/conversations/{thread_id}")
async def get_conversation_history(thread_id: str, user_id: str) -> ConversationHistory:
    try:
        return await get_conversation_history_from_agent(thread_id, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")


def run_server():
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001, 
        timeout_keep_alive=300,  # Keep idle connections alive for 120 seconds (default: 5)
        timeout_graceful_shutdown=30,  # Wait up to 30 seconds for graceful shutdown (default: 0)
    )