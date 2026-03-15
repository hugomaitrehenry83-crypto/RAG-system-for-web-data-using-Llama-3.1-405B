"""
retriever.py — Build a LangChain retriever from the vectorstore.
"""

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import AppConfig


def get_retriever(db: Chroma, config: AppConfig) -> VectorStoreRetriever:
    """
    Build a retriever from a ChromaDB instance.

    search_type:
      - "similarity" : standard cosine / L2 nearest neighbours
      - "mmr"        : Maximal Marginal Relevance — balances relevance
                       with diversity to avoid redundant chunks
    """
    search_kwargs = {"k": config.retriever.k}

    retriever = db.as_retriever(
        search_type=config.retriever.search_type,
        search_kwargs=search_kwargs,
    )

    print(
        f"Retriever configured: {config.retriever.search_type}, "
        f"k={config.retriever.k}"
    )
    return retriever
