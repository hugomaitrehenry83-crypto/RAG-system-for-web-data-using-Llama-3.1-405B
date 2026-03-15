"""
main.py — CLI entrypoint for the RAG Web System.

Usage:
  # Index URLs from config and ask one question
  python main.py --question "What is LangChain?"

  # Force reindex (re-fetch URLs)
  python main.py --reindex --question "What is RAG?"

  # Interactive Q&A loop
  python main.py --interactive

  # Use a custom config file
  python main.py --config my_config.yaml --question "..."
"""

import argparse
import sys

from src.config import load_config
from src.ingestion import WebIngestion
from src.embeddings import get_embeddings
from src.vectorstore import VectorStore
from src.retriever import get_retriever
from src.llm import get_llm
from src.chain import RAGChain


def build_pipeline(config, reindex: bool = False):
    """Initialize all RAG components and return a ready RAGChain."""
    # 1. Embeddings
    embeddings = get_embeddings(config)

    # 2. Vectorstore
    vs = VectorStore(config, embeddings)

    if reindex:
        ingestor = WebIngestion(config)
        chunks = ingestor.ingest()
        db = vs.build(chunks)
    else:
        try:
            db = vs.load()
        except FileNotFoundError:
            print("No existing vectorstore found — indexing URLs...")
            ingestor = WebIngestion(config)
            chunks = ingestor.ingest()
            db = vs.build(chunks)

    # 3. Retriever
    retriever = get_retriever(db, config)

    # 4. LLM
    llm = get_llm(config)

    # 5. Chain
    chain = RAGChain(retriever, llm, config)
    return chain


def interactive_loop(chain: RAGChain):
    """Run an interactive Q&A loop in the terminal."""
    print("\n" + "="*60)
    print("  RAG Web System — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        response = chain.ask(question)
        print(response)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="RAG Web System — Ask questions grounded in web content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    parser.add_argument(
        "--question", "-q", type=str,
        help="Question to ask the RAG system"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Launch interactive Q&A mode"
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Force re-fetching and re-indexing of all URLs"
    )

    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.print_help()
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Build the pipeline
    chain = build_pipeline(config, reindex=args.reindex)

    # Run
    if args.interactive:
        interactive_loop(chain)
    elif args.question:
        response = chain.ask(args.question)
        print(response)


if __name__ == "__main__":
    main()
