from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.conversation import Conversation
from app.models.message import Message
from typing import List, Dict, Any, Optional
import uuid
from functools import lru_cache


class ChatHistoryService:
    async def get_or_create_conversation(
        self, 
        user_id: str, 
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get existing conversation or create new one"""
        db = SessionLocal()
        try:
            if conversation_id:
                conv = db.query(Conversation).filter(
                    Conversation.id == uuid.UUID(conversation_id),
                    Conversation.user_id == uuid.UUID(user_id)
                ).first()
                
                if conv:
                    return {
                        "id": str(conv.id),
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat()
                    }
            
            # Create new conversation
            conv = Conversation(
                user_id=uuid.UUID(user_id),
                title="New Conversation"
            )
            db.add(conv)
            db.commit()
            db.refresh(conv)
            
            return {
                "id": str(conv.id),
                "title": conv.title,
                "created_at": conv.created_at.isoformat()
            }
        finally:
            db.close()
    
    async def save_message(
        self,
        user_id: str,
        conversation_id: Optional[str],
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Save a message to conversation"""
        db = SessionLocal()
        try:
            # Get or create conversation
            conv = await self.get_or_create_conversation(user_id, conversation_id)
            
            # Create message
            message = Message(
                conversation_id=uuid.UUID(conv["id"]),
                role=role,
                content=content,
                metadata=metadata or {}
            )
            db.add(message)
            db.commit()
            db.refresh(message)
            
            # Update conversation title if first message
            if role == "user":
                conv_obj = db.query(Conversation).filter(
                    Conversation.id == uuid.UUID(conv["id"])
                ).first()
                if conv_obj and conv_obj.title == "New Conversation":
                    # Use first few words as title
                    title = content[:50] + ("..." if len(content) > 50 else "")
                    conv_obj.title = title
                    db.commit()
            
            return {
                "id": str(message.id),
                "conversation_id": conv["id"],
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat()
            }
        finally:
            db.close()
    
    async def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's conversation history"""
        db = SessionLocal()
        try:
            # Get recent messages from all conversations
            messages = db.query(Message).join(
                Conversation, Conversation.id == Message.conversation_id
            ).filter(
                Conversation.user_id == uuid.UUID(user_id)
            ).order_by(
                Message.created_at.desc()
            ).limit(limit).all()
            
            # Reverse to get chronological order
            messages.reverse()
            
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "conversation_id": str(msg.conversation_id),
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
        finally:
            db.close()
    
    async def get_conversation_messages(
        self, 
        conversation_id: str, 
        user_id: str
    ) -> List[Dict]:
        """Get all messages in a conversation"""
        db = SessionLocal()
        try:
            messages = db.query(Message).join(
                Conversation, Conversation.id == Message.conversation_id
            ).filter(
                Message.conversation_id == uuid.UUID(conversation_id),
                Conversation.user_id == uuid.UUID(user_id)
            ).order_by(
                Message.created_at.asc()
            ).all()
            
            return [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
        finally:
            db.close()
    @lru_cache()
    def get_chat_history_service():
        return ChatHistoryService()