# Contributing

Contributions are welcome! Here's how to get started.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/rag-web-system.git
cd rag-web-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Project Structure

- `src/` — core modules (ingestion, embeddings, vectorstore, retriever, llm, chain)
- `tests/` — unit tests (one file per module)
- `notebooks/` — Jupyter demo notebook
- `config.yaml` — all tuneable parameters
- `main.py` — CLI entrypoint

## Adding a New LLM Provider

1. Edit `src/llm.py` and add a new `elif provider == "your_provider":` block
2. Add the dependency to `requirements.txt`
3. Document the new option in `config.yaml` and `README.md`

## Adding a New Embedding Provider

Same pattern in `src/embeddings.py`.

## Code Style

- PEP 8, max line length 100
- Docstrings on every public class and function
- Type hints encouraged
