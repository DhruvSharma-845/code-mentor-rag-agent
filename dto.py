from pydantic import BaseModel

class ChatRequest(BaseModel):
    messages: list[str]
    thread_id: str
    user_id: str

class MessageDetail(BaseModel):
    id: str
    role: str
    content: str

class ConversationHistory(BaseModel):
    thread_id: str
    messages: list[MessageDetail]
    user_id: str