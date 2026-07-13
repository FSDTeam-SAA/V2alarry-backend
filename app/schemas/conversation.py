from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class MessageBase(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict] = None

class MessageResponse(MessageBase):
    id: str
    created_at: datetime

class ConversationBase(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict] = None

class ConversationResponse(ConversationBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[MessageResponse]] = None