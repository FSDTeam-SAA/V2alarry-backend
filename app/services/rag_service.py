class RAGService:
    def __init__(self, vector_store, embedding_service, llm_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.workflow = create_chat_workflow()
    
    async def process_user_message(self, user_id: str, message: str, conversation_id: Optional[str] = None):
        state = {
            "user_id": user_id,
            "message": message,
            "conversation_id": conversation_id,
            "user_history": [],
            "retrieved_docs": [],
            "context": "",
            "response": "",
            "metadata": {}
        }
        
        result = await self.workflow.ainvoke(state)
        return result
    
    async def retrieve_relevant_docs(self, query: str, user_id: str, limit: int = 5):
        # Get query embedding
        query_embedding = await self.embedding_service.embed(query)
        
        # Search with filtering (only active documents)
        results = await self.vector_store.search(
            query_embedding,
            limit=limit,
            filter={"is_active": True}
        )
        
        return results


