"""
vectorstore.py — ChromaDB vectorstore: build, persist, and load.
"""

from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.config import AppConfig


class VectorStore:
    """
    Wrapper around ChromaDB for building and querying a persistent
    vector store from document chunks.
    """

    def __init__(self, config: AppConfig, embeddings: Embeddings):
        self.config = config
        self.embeddings = embeddings
        self.persist_dir = config.vectorstore.persist_dir
        self.collection_name = config.vectorstore.collection_name
        self._db: Chroma | None = None

    def build(self, chunks: List[Document]) -> Chroma:
        """
        Create a new ChromaDB collection from document chunks.
        Overwrites any existing collection with the same name.
        """
        print(f"\nBuilding vectorstore with {len(chunks)} chunks...")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        print(f"Vectorstore persisted at: {self.persist_dir}")
        return self._db

    def load(self) -> Chroma:
        """
        Load an existing ChromaDB collection from disk.
        Raises FileNotFoundError if it doesn't exist.
        """
        if not Path(self.persist_dir).exists():
            raise FileNotFoundError(
                f"No vectorstore found at '{self.persist_dir}'. "
                "Run with --reindex to build it first."
            )
        print(f"Loading vectorstore from: {self.persist_dir}")
        self._db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        return self._db

    def get_or_build(self, chunks: List[Document] = None) -> Chroma:
        """
        Load from disk if it exists; otherwise build from chunks.
        """
        if Path(self.persist_dir).exists():
            return self.load()
        if chunks is None:
            raise ValueError(
                "No existing vectorstore found and no chunks provided to build one."
            )
        return self.build(chunks)

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Run a quick similarity search (for debugging / testing)."""
        k = k or self.config.retriever.k
        if self._db is None:
            self.load()
        return self._db.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4):
        """Return (Document, score) tuples — lower score = more similar."""
        if self._db is None:
            self.load()
        return self._db.similarity_search_with_score(query, k=k)
