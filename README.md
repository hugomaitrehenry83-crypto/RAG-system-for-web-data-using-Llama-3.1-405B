#  RAG System for Web Data

A production-ready **Retrieval-Augmented Generation (RAG)** system that answers questions grounded in real web content — built with LangChain, ChromaDB, and open-source LLMs via Ollama.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  What is RAG?

Large Language Models have a knowledge cutoff and can hallucinate facts. RAG solves this by:

1. **Retrieving** relevant chunks from a custom knowledge base
2. **Augmenting** the LLM prompt with that context
3. **Generating** a grounded, accurate answer

```
User question
     │
     ▼
┌─────────────┐     similarity     ┌──────────────┐
│  Embedding  │ ─────────────────► │  ChromaDB    │
│   Model     │                    │  Vectorstore │
└─────────────┘                    └──────┬───────┘
                                          │ top-k chunks
                                          ▼
                                   ┌─────────────┐
                                   │  LLM        │  ──► Answer
                                   │  (Llama 3)  │
                                   └─────────────┘
```

---

##  Features

- **Modular architecture** — swap any component (LLM, embeddings, vectorstore)
- **Web ingestion** — load and index any public URL
- **Persistent vectorstore** — ChromaDB persists embeddings to disk
- **Configurable** — all parameters in a single `config.yaml`
- **CLI interface** — ask questions directly from the terminal
- **Tested** — unit tests for core components
- **100% open-source** — no paid API required (runs locally with Ollama)

---

##  Project Structure

```
rag-web-system/
├── src/
│   ├── ingestion.py       # Load & chunk web documents
│   ├── embeddings.py      # Embedding model wrapper
│   ├── vectorstore.py     # ChromaDB operations
│   ├── retriever.py       # Retrieval logic
│   ├── llm.py             # LLM wrapper (Ollama / OpenAI)
│   ├── chain.py           # Full RAG chain assembly
│   └── config.py          # Load configuration
├── tests/
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   └── test_chain.py
├── notebooks/
│   └── RAG_walkthrough.ipynb   # Step-by-step demo
├── data/
│   └── .gitkeep               # Vectorstore persisted here
├── config.yaml                # All tuneable parameters
├── main.py                    # CLI entrypoint
├── requirements.txt
└── README.md
```

---

##  Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/rag-web-system.git
cd rag-web-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama pull nomic-embed-text   # embedding model
```

### 3. Configure

Edit `config.yaml` to set your URLs, chunk size, model name, etc.

---

##  Usage

### Index URLs and ask a question

```bash
python main.py --question "What is LangChain?"
```

### Interactive mode

```bash
python main.py --interactive
```

### Use OpenAI instead of Ollama

```bash
OPENAI_API_KEY=sk-... python main.py --llm openai --question "What is RAG?"
```

---

##  How It Works — Step by Step

| Step | Module | Description |
|------|--------|-------------|
| 1. Ingest | `ingestion.py` | Fetch URLs → clean text → split into chunks |
| 2. Embed  | `embeddings.py` | Convert chunks to vector representations |
| 3. Store  | `vectorstore.py` | Persist vectors in ChromaDB |
| 4. Retrieve | `retriever.py` | Find top-k most relevant chunks for a query |
| 5. Generate | `chain.py` | Inject chunks into LLM prompt → get answer |

---

##  Configuration (`config.yaml`)

```yaml
urls:
  - https://en.wikipedia.org/wiki/LangChain
  - https://en.wikipedia.org/wiki/Retrieval-augmented_generation

chunking:
  chunk_size: 512
  chunk_overlap: 50

retriever:
  k: 4

llm:
  provider: ollama       # "ollama" or "openai"
  model: llama3
  temperature: 0.0
  max_tokens: 512

embeddings:
  provider: ollama
  model: nomic-embed-text

vectorstore:
  persist_dir: ./data/chroma_db
  collection_name: rag_collection
```

---

##  Running Tests

```bash
pytest tests/ -v
```

---

##  Example Output

```
Question: What is Retrieval-Augmented Generation?

Retrieved 4 chunks from knowledge base.

Answer:
Retrieval-Augmented Generation (RAG) is a technique that combines
information retrieval with text generation. Instead of relying solely
on the LLM's parametric knowledge, RAG first retrieves relevant
documents from a knowledge base, then conditions the LLM's response
on those documents — improving factual accuracy and reducing hallucination.

Sources:
 - https://en.wikipedia.org/wiki/Retrieval-augmented_generation
```

---

##  Extending the System

**Add a new data source:**
```python
from src.ingestion import WebIngestion
ingestor = WebIngestion(config)
ingestor.load_urls(["https://your-new-source.com"])
```

**Swap the LLM:**
```python
# In config.yaml
llm:
  provider: openai
  model: gpt-4o
```

---

##  References

- [LangChain documentation](https://python.langchain.com/)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [Ollama](https://ollama.com/)
- [RAG — Lewis et al., 2020](https://arxiv.org/abs/2005.11401)

---

