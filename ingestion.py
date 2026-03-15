"""
ingestion.py — Fetch web pages, clean text, and split into chunks.
"""

import re
import requests
from typing import List

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import AppConfig


class WebIngestion:
    """
    Handles loading web content and splitting it into chunks
    suitable for embedding and retrieval.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _fetch_url(self, url: str) -> str:
        """Fetch raw text content from a URL."""
        headers = {"User-Agent": "Mozilla/5.0 (RAG-system/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text

    def _clean_text(self, text: str) -> str:
        """
        Remove HTML tags, collapse whitespace, strip boilerplate.
        Works on both HTML pages and plain text files.
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove special HTML entities
        text = re.sub(r"&[a-z]+;", " ", text)
        # Collapse multiple whitespace / newlines into a single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def load_urls(self, urls: List[str] = None) -> List[Document]:
        """
        Fetch and return cleaned Document objects from a list of URLs.
        Falls back to config.urls if none provided.
        """
        urls = urls or self.config.urls
        documents: List[Document] = []

        for idx, url in enumerate(urls):
            print(f"[{idx+1}/{len(urls)}] Loading: {url}")
            try:
                raw_text = self._fetch_url(url)
                clean_text = self._clean_text(raw_text)
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": url, "doc_id": idx},
                )
                documents.append(doc)
                print(f"  ✓ Loaded {len(clean_text):,} chars")
            except Exception as e:
                print(f"  ✗ Failed to load {url}: {e}")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for embedding."""
        chunks = self.splitter.split_documents(documents)
        print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def ingest(self, urls: List[str] = None) -> List[Document]:
        """Full pipeline: fetch → clean → split."""
        documents = self.load_urls(urls)
        if not documents:
            raise ValueError("No documents were successfully loaded.")
        return self.split_documents(documents)
