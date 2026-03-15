"""
tests/test_retriever.py — Unit tests for the retriever and chain modules.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from src.config import AppConfig, RetrieverConfig, LLMConfig
from src.retriever import get_retriever
from src.chain import RAGChain, RAGResponse, _format_docs


@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.retriever = RetrieverConfig(k=3, search_type="similarity")
    cfg.llm = LLMConfig(provider="ollama", model="llama3")
    cfg.prompt_template = (
        "Context: {context}\nQuestion: {question}\nAnswer:"
    )
    return cfg


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.as_retriever.return_value = MagicMock()
    return db


def test_get_retriever_configures_k(mock_db, config):
    retriever = get_retriever(mock_db, config)
    mock_db.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


def test_format_docs():
    docs = [
        Document(page_content="First chunk.", metadata={}),
        Document(page_content="Second chunk.", metadata={}),
    ]
    result = _format_docs(docs)
    assert "First chunk." in result
    assert "Second chunk." in result
    assert "\n\n" in result


def test_rag_response_str():
    docs = [Document(page_content="ctx", metadata={"source": "https://example.com"})]
    resp = RAGResponse(
        question="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation.",
        source_documents=docs,
    )
    output = str(resp)
    assert "What is RAG?" in output
    assert "Retrieval-Augmented Generation" in output
    assert "https://example.com" in output


def test_rag_chain_ask(config):
    mock_retriever = MagicMock()
    sample_docs = [
        Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "https://test.com"})
    ]
    mock_retriever.invoke.return_value = sample_docs

    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "RAG is a technique combining retrieval and generation."

    rag = RAGChain(mock_retriever, mock_llm, config)
    rag._chain = mock_chain  # inject mock chain

    response = rag.ask("What is RAG?")

    assert isinstance(response, RAGResponse)
    assert response.question == "What is RAG?"
    assert "RAG is a technique" in response.answer
    assert response.source_documents == sample_docs
