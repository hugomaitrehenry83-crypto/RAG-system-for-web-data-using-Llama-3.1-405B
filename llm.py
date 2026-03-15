"""
llm.py — LLM factory.
Supports Ollama (local, free) and OpenAI.
"""

import os
from langchain_core.language_models import BaseLLM
from src.config import AppConfig


def get_llm(config: AppConfig) -> BaseLLM:
    """
    Return the appropriate LangChain LLM based on config.

    Providers:
      - "ollama"  : local model via Ollama (free, private)
                    requires: `ollama pull llama3`
      - "openai"  : OpenAI API (requires OPENAI_API_KEY env var)
    """
    provider = config.llm.provider.lower()

    if provider == "ollama":
        from langchain_ollama import OllamaLLM
        print(f"Using Ollama LLM: {config.llm.model}")
        return OllamaLLM(
            model=config.llm.model,
            temperature=config.llm.temperature,
            num_predict=config.llm.max_tokens,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )
        print(f"Using OpenAI LLM: {config.llm.model}")
        return ChatOpenAI(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            openai_api_key=api_key,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Choose 'ollama' or 'openai'."
        )
