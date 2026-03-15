"""
config.py — Load and expose configuration from config.yaml
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class RetrieverConfig:
    k: int = 4
    search_type: str = "similarity"


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "llama3"
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass
class EmbeddingsConfig:
    provider: str = "ollama"
    model: str = "nomic-embed-text"


@dataclass
class VectorstoreConfig:
    persist_dir: str = "./data/chroma_db"
    collection_name: str = "rag_collection"


@dataclass
class AppConfig:
    urls: List[str] = field(default_factory=list)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    vectorstore: VectorstoreConfig = field(default_factory=VectorstoreConfig)
    prompt_template: str = ""


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        urls=raw.get("urls", []),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        retriever=RetrieverConfig(**raw.get("retriever", {})),
        llm=LLMConfig(**raw.get("llm", {})),
        embeddings=EmbeddingsConfig(**raw.get("embeddings", {})),
        vectorstore=VectorstoreConfig(**raw.get("vectorstore", {})),
        prompt_template=raw.get("prompt_template", ""),
    )
