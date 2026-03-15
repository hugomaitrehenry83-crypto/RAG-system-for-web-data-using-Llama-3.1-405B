"""
embeddings.py — Embedding model factory.
Supports Ollama (local, free) and OpenAI.
"""

import os
from langchain_core.embeddings import Embeddings
from src.config import AppConfig


def get_embeddings(config: AppConfig) -> Embeddings:
    """
    Return the appropriate LangChain embedding model based on config.

    Providers:
      - "ollama"  : runs locally via Ollama (free, private)
                    requires: `ollama pull nomic-embed-text`
      - "openai"  : OpenAI API (requires OPENAI_API_KEY env var)
    """
    provider = config.embeddings.provider.lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        print(f"Using Ollama embeddings: {config.embeddings.model}")
        return OllamaEmbeddings(model=config.embeddings.model)

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )
        print(f"Using OpenAI embeddings: {config.embeddings.model}")
        return OpenAIEmbeddings(
            model=config.embeddings.model,
            openai_api_key=api_key,
        )

    else:
        raise ValueError(
            f"Unknown embeddings provider: '{provider}'. "
            "Choose 'ollama' or 'openai'."
        )
