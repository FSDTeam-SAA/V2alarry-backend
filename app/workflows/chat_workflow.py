# app/workflows/nodes.py
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.services.chat_history_service import ChatHistoryService
from app.core.llm import LLMService
from app.core.config import settings
from functools import lru_cache


class ChatState(TypedDict):
    user_id: str
    message: str
    conversation_id: Optional[str]
    user_history: List[Dict]
    retrieved_docs: List[Dict]
    context: str
    response: str
    metadata: Dict[str, Any]

class WorkflowNodes:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.chat_history_service = ChatHistoryService()
        self.llm_service = LLMService()
    
    async def retrieve_history(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve user's conversation history"""
        user_id = state.get("user_id")
        limit = 5  # Get last 5 messages for context
        
        history = await self.chat_history_service.get_user_history(user_id, limit)
        
        state["user_history"] = history
        state["metadata"]["history_retrieved"] = True
        
        return state
    
    async def retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents from vector DB"""
        message = state.get("message")
        user_id = state.get("user_id")
        
        # Generate embedding for the message
        query_embedding = await self.embedding_service.embed(message)
        
        # Search vector DB
        results = await self.vector_store.search(
            query_embedding,
            limit=settings.TOP_K_RETRIEVAL
        )
        
        # Extract content from results
        retrieved_docs = []
        for result in results:
            if result.get("payload"):
                retrieved_docs.append({
                    "content": result["payload"].get("content", ""),
                    "score": result.get("score", 0),
                    "document_id": result["payload"].get("document_id", ""),
                    "similarity": result.get("score", 0)
                })
        
        state["retrieved_docs"] = retrieved_docs
        state["metadata"]["docs_retrieved"] = len(retrieved_docs)
        
        return state
    
    async def generate_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate context from history and retrieved documents"""
        user_history = state.get("user_history", [])
        retrieved_docs = state.get("retrieved_docs", [])
        message = state.get("message")
        
        # Build context from documents
        doc_context = ""
        if retrieved_docs:
            doc_context = "Relevant information from knowledge base:\n\n"
            for i, doc in enumerate(retrieved_docs[:3], 1):  # Top 3 docs
                doc_context += f"[Source {i}] {doc['content']}\n\n"
        
        # Build history context
        history_context = ""
        if user_history:
            history_context = "Previous conversation:\n\n"
            for msg in user_history[-5:]:  # Last 5 messages
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_context += f"{role}: {msg.get('content')}\n"
            history_context += "\n"
        
        # Combine contexts
        full_context = f"{history_context}{doc_context}"
        
        state["context"] = full_context
        state["metadata"]["context_length"] = len(full_context)
        
        return state
    
    async def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI response using context"""
        message = state.get("message")
        context = state.get("context", "")
        user_history = state.get("user_history", [])
        
        # Build prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
        Rules:
        1. Use the context to answer questions accurately
        2. If context is provided, base your answer on it
        3. If no context, use your general knowledge
        4. Be concise and helpful
        5. Reference the source when using specific information from context
        
        Context:
        {context}
        
        User history:
        {history}
        """
        
        # Format history for prompt
        history_text = ""
        if user_history:
            last_messages = user_history[-3:]  # Last 3 exchanges
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_messages])
        
        formatted_prompt = system_prompt.format(
            context=context or "No specific context provided.",
            history=history_text or "No previous conversation."
        )
        
        # Generate response
        response = await self.llm_service.generate(
            system_prompt=formatted_prompt,
            user_message=f"User question: {message}"
        )
        
        state["response"] = response
        state["metadata"]["response_length"] = len(response)
        
        return state
    
    async def save_conversation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Save conversation to database"""
        user_id = state.get("user_id")
        message = state.get("message")
        response = state.get("response")
        conversation_id = state.get("conversation_id")
        metadata = state.get("metadata", {})
        
        # Save user message
        user_msg = await self.chat_history_service.save_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role="user",
            content=message,
            metadata=metadata
        )
        
        # Save assistant response
        assistant_msg = await self.chat_history_service.save_message(
            user_id=user_id,
            conversation_id=conversation_id or user_msg["conversation_id"],
            role="assistant",
            content=response,
            metadata={
                "retrieved_docs": metadata.get("docs_retrieved", 0),
                "context_length": metadata.get("context_length", 0)
            }
        )
        
        state["conversation_id"] = user_msg["conversation_id"]
        state["metadata"]["saved"] = True
        
        return state
    @lru_cache()
    def get_chat_workflow():
        return ChatWorkflow()


class ChatWorkflow:
    def __init__(self):
        self.nodes = WorkflowNodes()
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _create_workflow(self):
        workflow = StateGraph(ChatState)
        
        workflow.add_node("retrieve_history", self.nodes.retrieve_history)
        workflow.add_node("retrieve_documents", self.nodes.retrieve_documents)
        workflow.add_node("generate_context", self.nodes.generate_context)
        workflow.add_node("generate_response", self.nodes.generate_response)
        workflow.add_node("save_conversation", self.nodes.save_conversation)
        
        workflow.set_entry_point("retrieve_history")
        workflow.add_edge("retrieve_history", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_context")
        workflow.add_edge("generate_context", "generate_response")
        workflow.add_edge("generate_response", "save_conversation")
        workflow.add_edge("save_conversation", END)
        
        return workflow
    
    async def process_message(
        self,
        user_id: str,
        message: str,
        conversation_id: Optional[str] = None,
    ):
        """Process user message through the workflow"""
        initial_state = ChatState(
            user_id=user_id,
            message=message,
            conversation_id=conversation_id,
            user_history=[],
            retrieved_docs=[],
            context="",
            response="",
            metadata={},
        )
        
        return await self.app.ainvoke(initial_state)
