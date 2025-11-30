import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 

from contextlib import asynccontextmanager

from fastapi.responses import StreamingResponse

from a2a_manager import get_agent_card
from app_bootstrapper import bootstrap_app, destroy_app
from conversation_service import chat_with_agent, chat_with_agent_stream_generator, get_all_conversation_ids, get_conversation_history_from_agent
from dto import A2AErrorCode, A2ARequest, A2AResponse, A2AStreamCompletionResponse, AgentCard, ChatRequest, ConversationHistory
from utils import MessageConverter, getValueFromConfig

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

@app.get("/.well-known/agent-card.json")
async def get_agent_card_endpoint() -> AgentCard:
    """
    Agent Card endpoint for A2A discovery.
    Returns information about this agent's capabilities and A2A endpoint.
    
    Args:
        assistant_id: Optional assistant ID for multi-agent systems
    """
    return get_agent_card()

@app.post("/a2a/{assistant_id}")
async def a2a_endpoint(assistant_id: str, request: A2ARequest) -> A2AResponse:
    try:
        # Validate assistant ID
        config = getValueFromConfig("a2a", "card_details")
        if assistant_id != config["agent_id"]:
            return A2AResponse.error_response(
                request_id=request.id,
                code=A2AErrorCode.AGENT_NOT_FOUND,
                message=f"Agent '{assistant_id}' not found",
                data={"available_agents": [config["agent_id"]]}
            )
        
        # Check if method is supported
        if request.method not in ["invoke", "stream"]:
            return A2AResponse.error_response(
                request_id=request.id,
                code=A2AErrorCode.METHOD_NOT_FOUND,
                message=f"Method '{request.method}' not supported. Use 'invoke' or 'stream'."
            )
        
        # Convert A2A messages to LangChain format
        a2a_messages = request.messages
    
        
        # Invoke the agent
        result = await chat_with_agent(request.thread_id, a2a_messages, request.user_id)
    
        
        # Create successful response
        return A2AResponse.success(
            request_id=request.id,
            messages=result.messages,
            thread_id=request.thread_id,
            user_id=request.user_id,
            metadata={
                "message_count": len(result.messages),
                "agent_id": assistant_id
            }
        )
        
    except ValueError as e:
        # Validation errors
        return A2AResponse.error_response(
            request_id=request.id,
            code=A2AErrorCode.INVALID_PARAMS,
            message=str(e)
        )
    except Exception as e:
        # Internal errors
        print(f"A2A endpoint error: {e}")
        import traceback
        traceback.print_exc()
        
        return A2AResponse.error_response(
            request_id=request.id,
            code=A2AErrorCode.INTERNAL_ERROR,
            message="Internal server error",
            data={"error": str(e)}
        )

@app.post("/a2a/{assistant_id}/stream")
async def a2a_stream_endpoint(assistant_id: str, request: A2ARequest):
    try:
        # Validate assistant ID
        config = getValueFromConfig("a2a", "card_details")
        if assistant_id != config["agent_id"]:
            error_response = A2AResponse.error_response(
                request_id=request.id,
                code=A2AErrorCode.AGENT_NOT_FOUND,
                message=f"Agent '{assistant_id}' not found"
            )
            async def error_stream():
                yield f"data: {error_response.model_dump_json()}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        
        
        # Create streaming generator
        async def a2a_stream_generator(request: A2ARequest):
            """Generator that yields A2A formatted streaming responses"""
            try:
                from conversation_service import chat_with_agent_stream_generator
                
                # Stream from internal agent
                async for chunk in chat_with_agent_stream_generator(
                    thread_id=request.thread_id,
                    input_messages=request.messages,
                    user_id=request.user_id
                ):
                    a2a_response = A2AResponse.success(
                        request_id=request.id,
                        messages=chunk.messages,
                        thread_id=request.thread_id,
                        user_id=request.user_id,
                        metadata={
                            "message_count": len(chunk.messages),
                            "agent_id": assistant_id
                        }
                    )
                    yield f"data: {a2a_response.model_dump_json()}\n\n"
                
                # Send completion message
                completion = A2AStreamCompletionResponse(
                    id=request.id,
                    result={
                        "status": "complete",
                        "thread_id": request.thread_id
                    }
                )
                yield f"data: {completion.model_dump_json()}\n\n"
                
            except Exception as e:
                error_response = A2AResponse.error_response(
                    request_id=request.id,
                    code=A2AErrorCode.INTERNAL_ERROR,
                    message=str(e)
                )
                yield f"data: {error_response.model_dump_json()}\n\n"
        
        return StreamingResponse(
            a2a_stream_generator(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        print(f"A2A stream endpoint error: {e}")
        error_response = A2AResponse.error_response(
            request_id=request.id if hasattr(request, 'id') else "unknown",
            code=A2AErrorCode.INTERNAL_ERROR,
            message=str(e)
        )
        async def error_stream():
            yield f"data: {error_response.model_dump_json()}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/api/code-mentor/chat")
async def universal_agent_chat(request: ChatRequest) -> ConversationHistory:
    return await chat_with_agent(request.thread_id, request.messages, request.user_id)

@app.post("/api/code-mentor/chat/stream")
async def universal_agent_chat_stream(request: ChatRequest) -> StreamingResponse:
    async def chat_stream_generator(request: ChatRequest):
        try:
            async for chunk in chat_with_agent_stream_generator(
                thread_id=request.thread_id,
                input_messages=request.messages,
                user_id=request.user_id
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        chat_stream_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            }
        )

@app.get("/api/code-mentor/conversations")
async def list_all_conversations(user_id: str) -> dict:
    # try:
        thread_ids = await get_all_conversation_ids(user_id)
        return {"threads": thread_ids, "count": len(thread_ids)}
    # except Exception as e:
        # raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@app.get("/api/code-mentor/conversations/{thread_id}")
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