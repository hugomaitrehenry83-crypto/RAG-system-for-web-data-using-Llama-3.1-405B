"""
chain.py — Assemble the full RAG chain and run queries.
"""

from typing import List
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models import BaseLLM
from langchain.schema import Document

from src.config import AppConfig


@dataclass
class RAGResponse:
    """Structured response from the RAG chain."""
    question: str
    answer: str
    source_documents: List[Document]

    def __str__(self) -> str:
        sources = "\n".join(
            f"  - {doc.metadata.get('source', 'unknown')}"
            for doc in self.source_documents
        )
        return (
            f"\nQuestion: {self.question}\n"
            f"\nAnswer:\n{self.answer}\n"
            f"\nSources:\n{sources}"
        )


def _format_docs(docs: List[Document]) -> str:
    """Concatenate document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """
    Assembles and runs the full RAG pipeline:
      retriever → prompt → LLM → output parser
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: BaseLLM,
        config: AppConfig,
    ):
        self.retriever = retriever
        self.llm = llm
        self.config = config
        self._chain = self._build_chain()

    def _build_chain(self):
        """Build the LangChain LCEL chain."""
        prompt = ChatPromptTemplate.from_template(
            self.config.prompt_template
        )

        chain = (
            {
                "context": self.retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def ask(self, question: str) -> RAGResponse:
        """
        Ask a question and return a structured RAGResponse
        (answer + source documents).
        """
        # Retrieve documents separately for attribution
        source_docs = self.retriever.invoke(question)

        # Run the full chain for the answer
        answer = self._chain.invoke(question)

        return RAGResponse(
            question=question,
            answer=answer.strip(),
            source_documents=source_docs,
        )
