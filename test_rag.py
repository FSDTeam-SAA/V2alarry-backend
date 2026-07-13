# test_rag.py
import asyncio
from app.workflows.chat_workflow import ChatWorkflow
import uuid

async def test_rag():
    workflow = ChatWorkflow()
    
    # Test with a user
    user_id = str(uuid.uuid4())
    
    print("🤖 Testing RAG Workflow\n")
    print("=" * 50)
    
    # First message
    result1 = await workflow.process_message(
        user_id=user_id,
        message="What is AI?",
        conversation_id=None
    )
    
    print(f"User: {result1['message']}")
    print(f"Assistant: {result1['response']}")
    print(f"Retrieved docs: {result1['metadata'].get('docs_retrieved', 0)}")
    print(f"Context length: {result1['metadata'].get('context_length', 0)}")
    print("=" * 50)
    
    # Follow-up message (with history)
    result2 = await workflow.process_message(
        user_id=user_id,
        message="Tell me more about machine learning",
        conversation_id=result1['conversation_id']
    )
    
    print(f"User: {result2['message']}")
    print(f"Assistant: {result2['response']}")
    print(f"Retrieved docs: {result2['metadata'].get('docs_retrieved', 0)}")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_rag())