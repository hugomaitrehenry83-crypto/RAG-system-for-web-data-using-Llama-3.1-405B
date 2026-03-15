"""
tests/test_ingestion.py — Unit tests for the ingestion module.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from src.config import AppConfig, ChunkingConfig
from src.ingestion import WebIngestion


@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.urls = ["https://example.com/page1", "https://example.com/page2"]
    cfg.chunking = ChunkingConfig(chunk_size=100, chunk_overlap=10)
    return cfg


def test_clean_text_removes_html(config):
    ingestor = WebIngestion(config)
    raw = "<html><body><h1>Hello World</h1><p>Some text here.</p></body></html>"
    clean = ingestor._clean_text(raw)
    assert "<html>" not in clean
    assert "Hello World" in clean
    assert "Some text here." in clean


def test_clean_text_collapses_whitespace(config):
    ingestor = WebIngestion(config)
    raw = "Hello   \n\n\n   World"
    clean = ingestor._clean_text(raw)
    assert clean == "Hello World"


@patch("src.ingestion.requests.get")
def test_load_urls_success(mock_get, config):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<p>This is a test document with enough content.</p>"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    ingestor = WebIngestion(config)
    docs = ingestor.load_urls(["https://example.com/test"])

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "test document" in docs[0].page_content
    assert docs[0].metadata["source"] == "https://example.com/test"


@patch("src.ingestion.requests.get")
def test_load_urls_handles_failure(mock_get, config):
    mock_get.side_effect = Exception("Connection error")
    ingestor = WebIngestion(config)
    docs = ingestor.load_urls(["https://unreachable.example.com"])
    # Should not raise — failed URLs are skipped
    assert docs == []


def test_split_documents(config):
    ingestor = WebIngestion(config)
    long_text = "word " * 300  # ~1500 chars → should produce multiple chunks
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = ingestor.split_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= config.chunking.chunk_size + 50
