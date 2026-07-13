# app/workflows/chat_workflow.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from app.workflows.nodes import WorkflowNodes

class ChatState(TypedDict):
    user_id: str
    message: str
    conversation_id: Optional[str]
    user_history: List[Dict]
    retrieved_docs: List[Dict]
    context: str
    response: str
    metadata: Dict[str, Any]

class ChatWorkflow:
    def __init__(self):
        self.nodes = WorkflowNodes()
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _create_workflow(self):
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("retrieve_history", self.nodes.retrieve_history)
        workflow.add_node("retrieve_documents", self.nodes.retrieve_documents)
        workflow.add_node("generate_context", self.nodes.generate_context)
        workflow.add_node("generate_response", self.nodes.generate_response)
        workflow.add_node("save_conversation", self.nodes.save_conversation)
        
        # Add edges
        workflow.set_entry_point("retrieve_history")
        workflow.add_edge("retrieve_history", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_context")
        workflow.add_edge("generate_context", "generate_response")
        workflow.add_edge("generate_response", "save_conversation")
        workflow.add_edge("save_conversation", END)
        
        return workflow
    
    async def process_message(self, user_id: str, message: str, conversation_id: Optional[str] = None):
        """Process user message through the workflow"""
        initial_state = ChatState(
            user_id=user_id,
            message=message,
            conversation_id=conversation_id,
            user_history=[],
            retrieved_docs=[],
            context="",
            response="",
            metadata={}
        )
        
        result = await self.app.ainvoke(initial_state)
        return result