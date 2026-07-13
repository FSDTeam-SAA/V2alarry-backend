# app/core/llm.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import settings
import os

class LLMService:
    def __init__(self):
        # Initialize OpenAI (or use open-source)
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # For local LLM (Ollama), use:
        # from langchain_community.llms import Ollama
        # self.llm = Ollama(model="llama3.2")
    
    async def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate response using LLM"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content
