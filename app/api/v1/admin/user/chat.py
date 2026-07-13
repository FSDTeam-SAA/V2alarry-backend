@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Send a message and get AI response with context"""
    response = await rag_service.process_user_message(
        user_id=current_user.id,
        message=request.message,
        conversation_id=request.conversation_id
    )
    return response

@router.get("/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user)
):
    """Get user's conversation history"""
    pass

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all messages in a conversation"""
    pass