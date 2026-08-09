from typing import Dict, Any, List, Optional, TypedDict, AsyncIterator
from langgraph.graph import StateGraph, END
from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.services.chat_history_service import ChatHistoryService
from app.core.llm import LLMService
from app.core.config import settings
from app.core.cache import cache_get, cache_set, make_cache_key
import asyncio
import json


class ChatState(TypedDict):
    user_id: str
    message: str
    conversation_id: Optional[str]
    user_history: List[Dict]
    retrieved_docs: List[Dict]
    context: str
    response: str
    metadata: Dict[str, Any]


SYSTEM_PROMPT = """You are a precise, knowledgeable AI assistant. Your goal is to provide accurate, well-structured answers.

RULES:
1. ANSWER FROM CONTEXT FIRST: If relevant information is provided in the context, use it as your primary source. Cite sources using [Source N] notation.
2. BE PRECISE: Give direct, specific answers. Avoid vague or generic responses.
3. HANDLE MISSING CONTEXT: If the context does not contain enough information, clearly state what is missing and then provide your best general knowledge answer — but explicitly label it as general knowledge.
4. DO NOT HALLUCINATE: Never fabricate facts, statistics, or source names. If you are unsure, say so.
5. STRUCTURE YOUR RESPONSES: Use headers, bullet points, or numbered lists for clarity when answering complex questions.
6. MAINTAIN CONVERSATION FLOW: Reference previous messages when relevant to provide coherent, contextual answers.
7. BE CONCISE: Prefer shorter, focused answers. Only elaborate when the question demands depth.
8. IF ASKED ABOUT SOMETHING OUTSIDE YOUR KNOWLEDGE AND CONTEXT: Say "I don't have enough information to answer this accurately" rather than guessing."""


class WorkflowNodes:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.chat_history_service = ChatHistoryService()
        self.llm_service = LLMService()

    async def retrieve_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel retrieval of history and documents."""
        user_id = state.get("user_id")
        message = state.get("message")
        conversation_id = state.get("conversation_id")

        history_task = asyncio.create_task(
            self._fetch_history(user_id, conversation_id)
        )
        docs_task = asyncio.create_task(
            self._fetch_documents(message)
        )

        user_history, retrieved_docs = await asyncio.gather(
            history_task, docs_task
        )

        state["user_history"] = user_history
        state["retrieved_docs"] = retrieved_docs
        state["metadata"]["history_retrieved"] = len(user_history)
        state["metadata"]["docs_retrieved"] = len(retrieved_docs)

        return state

    async def _fetch_history(
        self, user_id: str, conversation_id: Optional[str]
    ) -> List[Dict]:
        cache_key = f"hist:{make_cache_key(user_id, conversation_id or 'new')}"
        cached = await cache_get(cache_key)
        if cached is not None:
            return cached

        if conversation_id:
            history = await self.chat_history_service.get_conversation_history(
                conversation_id=conversation_id,
                user_id=user_id,
                limit=settings.HISTORY_LIMIT,
            )
        else:
            history = []

        await cache_set(cache_key, history, ttl=settings.CACHE_HISTORY_TTL)
        return history

    async def _fetch_documents(self, message: str) -> List[Dict]:
        query_embedding = await self.embedding_service.embed(message)

        results = await self.vector_store.search(
            query_embedding,
            limit=settings.TOP_K_RETRIEVAL,
        )

        retrieved_docs = []
        for result in results:
            score = result.get("score", 0)
            if score < settings.SIMILARITY_THRESHOLD:
                continue
            if result.get("payload"):
                retrieved_docs.append({
                    "content": result["payload"].get("content", ""),
                    "score": score,
                    "document_id": result["payload"].get("document_id", ""),
                    "similarity": score,
                })

        return retrieved_docs

    async def generate_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_history = state.get("user_history", [])
        retrieved_docs = state.get("retrieved_docs", [])

        doc_context = ""
        if retrieved_docs:
            doc_context = "Relevant information from knowledge base:\n\n"
            for i, doc in enumerate(retrieved_docs[: settings.CONTEXT_DOC_COUNT], 1):
                doc_context += f"[Source {i}] (relevance: {doc['score']:.2f}) {doc['content']}\n\n"

        history_context = ""
        if user_history:
            history_context = "Previous conversation:\n\n"
            for msg in user_history[-settings.HISTORY_LIMIT:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_context += f"{role}: {msg.get('content')}\n"
            history_context += "\n"

        full_context = f"{history_context}{doc_context}"

        state["context"] = full_context
        state["metadata"]["context_length"] = len(full_context)

        return state

    async def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        message = state.get("message")
        context = state.get("context", "")

        formatted_prompt = SYSTEM_PROMPT + "\n\nContext:\n" + (
            context if context else "No specific context provided."
        )

        response = await self.llm_service.generate(
            system_prompt=formatted_prompt,
            user_message=message,
        )

        state["response"] = response
        state["metadata"]["response_length"] = len(response)

        return state

    async def generate_response_stream(
        self, state: Dict[str, Any]
    ) -> AsyncIterator[str]:
        message = state.get("message")
        context = state.get("context", "")

        formatted_prompt = SYSTEM_PROMPT + "\n\nContext:\n" + (
            context if context else "No specific context provided."
        )

        full_response = ""
        async for token in self.llm_service.generate_stream(
            system_prompt=formatted_prompt,
            user_message=message,
        ):
            full_response += token
            yield json.dumps({"type": "token", "content": token}) + "\n"

        state["response"] = full_response
        state["metadata"]["response_length"] = len(full_response)

    async def save_conversation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_id = state.get("user_id")
        message = state.get("message")
        response = state.get("response")
        conversation_id = state.get("conversation_id")
        metadata = state.get("metadata", {})

        result = await self.chat_history_service.save_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=message,
            assistant_response=response,
            metadata=metadata,
        )

        state["conversation_id"] = result["conversation_id"]
        state["metadata"]["saved"] = True
        state["metadata"]["message_id"] = result["id"]
        state["metadata"]["user_message_id"] = result["user_message_id"]
        state["metadata"]["assistant_message_id"] = result["assistant_message_id"]

        await self._invalidate_history_cache(user_id, conversation_id)

        return state

    async def _invalidate_history_cache(
        self, user_id: str, conversation_id: Optional[str]
    ):
        pattern = f"hist:{make_cache_key(user_id, '*')}"
        await cache_delete_safe(pattern)

    async def query_rewriting(self, state: Dict[str, Any]) -> Dict[str, Any]:
        message = state.get("message")
        user_history = state.get("user_history", [])

        if not user_history:
            state["rewritten_query"] = message
            state["metadata"]["query_rewritten"] = False
            return state

        recent_context = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in user_history[-3:]
        )

        rewrite_prompt = f"""Given the conversation history and the latest user message, rewrite the latest message as a standalone search query that captures the full intent, including any references to previous context.

Conversation history:
{recent_context}

Latest user message: {message}

Output ONLY the rewritten query text, nothing else."""

        rewritten = await self.llm_service.generate(
            system_prompt="You are a query rewriting assistant. Output only the rewritten query.",
            user_message=rewrite_prompt,
        )

        state["rewritten_query"] = rewritten.strip()
        state["metadata"]["query_rewritten"] = True

        return state


async def cache_delete_safe(pattern: str):
    try:
        from app.core.cache import cache_delete_pattern
        await cache_delete_pattern(pattern)
    except Exception:
        pass


class ChatWorkflow:
    def __init__(self):
        self.nodes = WorkflowNodes()
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_workflow(self):
        workflow = StateGraph(ChatState)

        workflow.add_node("retrieve_data", self.nodes.retrieve_data)
        workflow.add_node("generate_context", self.nodes.generate_context)
        workflow.add_node("generate_response", self.nodes.generate_response)
        workflow.add_node("save_conversation", self.nodes.save_conversation)

        workflow.set_entry_point("retrieve_data")
        workflow.add_edge("retrieve_data", "generate_context")
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

    async def process_message_stream(
        self,
        user_id: str,
        message: str,
        conversation_id: Optional[str] = None,
    ):
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

        nodes = self.nodes

        yield json.dumps({"type": "status", "status": "accepted"})
        yield json.dumps({"type": "status", "status": "retrieving_context"})
        state = await nodes.retrieve_data(initial_state)
        yield json.dumps({"type": "status", "status": "building_context"})
        state = await nodes.generate_context(state)

        yield json.dumps({"type": "status", "status": "generating_response"})
        async for event in nodes.generate_response_stream(state):
            yield event

        state = await nodes.save_conversation(state)
        yield json.dumps(
            {
                "type": "done",
                "conversation_id": state["conversation_id"],
                "user_message_id": state["metadata"]["user_message_id"],
                "assistant_message_id": state["metadata"]["assistant_message_id"],
                "persisted": True,
            }
        )


_workflow_instance: Optional[ChatWorkflow] = None


def get_chat_workflow() -> ChatWorkflow:
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = ChatWorkflow()
    return _workflow_instance
