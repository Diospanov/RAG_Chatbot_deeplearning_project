"""
generation/generator.py
-----------------------
LLM generation with context grounding.

Supports three providers:
  - anthropic  (Claude, default)
  - openai     (GPT)
  - ollama     (local models via Ollama)

The provider is selected via the LLM_PROVIDER env var or constructor arg.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from generation.prompts import build_messages, REFUSAL_PHRASE
from retrieval.retriever import Retriever, format_context

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    End-to-end RAG: retrieve relevant chunks, build a grounded prompt,
    call the LLM, and return the answer with source citations.
    """

    def __init__(
        self,
        provider: str | None = None,
        top_k: int | None = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.retriever = Retriever(top_k=top_k)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        history: Optional[List[tuple]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG.

        Returns:
            {
                "answer": str,
                "sources": list[dict],   # retrieved chunks with metadata
                "context": str,          # formatted context sent to LLM
                "is_refusal": bool,
            }
        """
        if not self.retriever.is_ready():
            return {
                "answer": "The knowledge base is empty. Please ingest documents first.",
                "sources": [],
                "context": "",
                "is_refusal": True,
            }

        # 1. Retrieve
        chunks = self.retriever.retrieve(query)
        context = format_context(chunks)

        # 2. Generate
        system_prompt, messages = build_messages(query, context, history)
        answer = self._call_llm(system_prompt, messages)

        # 3. Detect refusals
        is_refusal = REFUSAL_PHRASE.lower() in answer.lower()

        return {
            "answer": answer,
            "sources": chunks,
            "context": context,
            "is_refusal": is_refusal,
        }

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, messages: List[Dict]) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, messages)
        elif self.provider == "openai":
            return self._call_openai(system_prompt, messages)
        elif self.provider == "ollama":
            return self._call_ollama(system_prompt, messages)
        elif self.provider == "deepseek":
            return self._call_deepseek(system_prompt, messages)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider!r}")
        

    # --- Anthropic ---

    def _call_anthropic(self, system_prompt: str, messages: List[Dict]) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    # --- OpenAI ---

    def _call_openai(self, system_prompt: str, messages: List[Dict]) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content

    # --- Ollama ---

    def _call_ollama(self, system_prompt: str, messages: List[Dict]) -> str:
        import requests

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        resp = requests.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": full_messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    
    # --- DeepSeek ---
    
    def _call_deepseek(self, system_prompt: str, messages: list) -> str:
        from openai import OpenAI 

        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content
