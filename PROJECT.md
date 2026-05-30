# RAG Corpus — Project Documentation

A fully local Retrieval-Augmented Generation (RAG) system built with LangChain, Chroma, and Ollama.
No cloud API keys required. All inference runs on your own machine.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Corpus](#corpus)
5. [Architecture](#architecture)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Setup & Running](#setup--running)
9. [How It Works — Step by Step](#how-it-works--step-by-step)
10. [Key Design Decisions](#key-design-decisions)
11. [Known Limitations](#known-limitations)

---

## Overview

This project lets you ask natural language questions against a private document corpus and receive
grounded answers with source citations. The pipeline:

1. **Ingest** — load your PDFs and CSV files, split them into chunks, embed each chunk with a local
   embedding model, and store the vectors in a persistent Chroma database.
2. **Ask** — embed the user's question, retrieve the most relevant chunks from Chroma, pass them as
   context to a local LLM, and return the answer together with the source snippets.

Everything runs locally via **Ollama**. No data leaves the machine.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.13 |
| Web framework | FastAPI | 0.136.3 |
| ASGI server | Uvicorn | latest |
| RAG orchestration | LangChain | 1.3.2 |
| Chroma integration | langchain-chroma | 1.1.0 |
| Ollama integration | langchain-ollama | 1.1.0 |
| Vector store | Chroma (embedded) | 1.5.9 |
| LLM (local) | Ollama — llama3.2 | 2.0 GB |
| Embeddings (local) | Ollama — nomic-embed-text | 274 MB |
| PDF loading | pypdf (via LangChain) | latest |
| CSV loading | LangChain CSVLoader | — |
| Frontend | Vanilla HTML/CSS/JS | single file |

---

## Project Structure

```
DSAI_04_Rag_BenAu/
│
├── backend/
│   ├── __init__.py       # Package marker
│   ├── main.py           # FastAPI app — all HTTP endpoints
│   ├── ingest.py         # Document loading, chunking, embedding, Chroma upsert
│   └── rag.py            # LCEL retrieval chain + ChatOllama answer generation
│
├── frontend/
│   └── index.html        # Single-page UI (Ingest tab + Ask tab)
│
├── data/                 # Drop corpus files here (PDF, CSV)
├── chroma_db/            # Chroma persistent store (auto-created on first ingest)
│
├── .env                  # Runtime configuration (models, paths, chunk settings)
├── requirements.txt      # Python dependencies
├── start.sh              # One-command startup script
└── PROJECT.md            # This file
```

---

## Corpus

The current corpus covers **CPF (Central Provident Fund)** — Singapore's mandatory savings scheme
for retirement, housing, and healthcare.

| File | Type | Description |
|---|---|---|
| `CPF FAQ.pdf` | PDF | Frequently asked questions on CPF |
| `CPFAllocationRatesfromJanuary2026.pdf` | PDF | CPF allocation rates effective Jan 2026 |
| `CPFallocationrates.csv` | CSV | Tabular allocation rate data |
| `Guide to understanding CPF LIFE.pdf` | PDF | CPF LIFE annuity scheme guide |
| `HANDY CPF resources for retirement planning.pdf` | PDF | Retirement planning reference |

After ingestion, the corpus produces **186 chunks** stored in Chroma.

To swap in a different corpus, replace the files in `data/` and click **Ingest Documents** again.
The old collection is automatically deleted and rebuilt from scratch.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Browser UI                         │
│              frontend/index.html                        │
│         ┌──────────────┬──────────────┐                 │
│         │  Ingest Tab  │   Ask Tab    │                 │
└─────────┼──────────────┼──────────────┼─────────────────┘
          │ POST /ingest │ POST /ask    │
          ▼              ▼              │
┌─────────────────────────────────────────────────────────┐
│                  FastAPI (main.py)                      │
│   GET  /              → serves index.html               │
│   GET  /api/status    → document count                  │
│   GET  /api/files     → list data/ directory            │
│   POST /api/ingest    → triggers ingest pipeline        │
│   POST /api/ask       → triggers RAG query              │
└────────────────────────────┬────────────────────────────┘
                             │
          ┌──────────────────┴────────────────────┐
          │                                       │
          ▼                                       ▼
┌──────────────────────┐             ┌────────────────────────┐
│    ingest.py         │             │      rag.py            │
│                      │             │                        │
│  PyPDFLoader         │             │  get_vectorstore()     │
│  CSVLoader           │             │  → Chroma retriever    │
│      ↓               │             │      ↓                 │
│  RecursiveCharacter  │             │  LCEL chain:           │
│  TextSplitter        │             │  {context, question}   │
│  (1000 / 200 overlap)│             │   | ChatPromptTemplate │
│      ↓               │             │   | ChatOllama         │
│  OllamaEmbeddings    │             │   | StrOutputParser    │
│  (nomic-embed-text)  │             │      ↓                 │
│      ↓               │             │  answer + sources      │
│  Chroma.from_docs()  │             └────────────────────────┘
│  → chroma_db/        │                        │
└──────────────────────┘                        │
          │                                     │
          └──────────────┬──────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Chroma DB          │
              │  (chroma_db/)       │
              │  collection:        │
              │  "rag_corpus"       │
              │  186 chunks         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Ollama             │
              │  localhost:11434    │
              │                    │
              │  nomic-embed-text  │  ← embeddings
              │  llama3.2          │  ← chat completions
              └─────────────────────┘
```

---

## API Reference

All endpoints are served from `http://localhost:8000`.

### `GET /`
Serves the HTML frontend (`frontend/index.html`).

---

### `GET /api/status`
Returns the current state of the vector store.

**Response**
```json
{
  "has_documents": true,
  "document_count": 186
}
```

---

### `GET /api/files`
Lists all supported files found in the `data/` directory.

**Response**
```json
{
  "files": [
    { "name": "CPF FAQ.pdf", "size": 204800, "type": "pdf" },
    { "name": "CPFallocationrates.csv", "size": 4096, "type": "csv" }
  ]
}
```

---

### `POST /api/ingest`
Scans `data/`, loads all PDF and CSV files, splits them into chunks, embeds them with
`nomic-embed-text`, and stores them in Chroma. Any existing collection is deleted first.

**Response (success)**
```json
{
  "status": "success",
  "documents_ingested": 186,
  "files_processed": ["CPF FAQ.pdf", "CPFallocationrates.csv"],
  "errors": []
}
```

**Response (error — HTTP 400)**
```json
{ "detail": "No supported files found in the data directory." }
```

---

### `POST /api/ask`
Retrieves the top-K relevant chunks for the question and generates an answer using `llama3.2`.

**Request body**
```json
{ "question": "What is the CPF LIFE standard plan?" }
```

**Response**
```json
{
  "answer": "CPF LIFE Standard Plan provides higher monthly payouts...",
  "sources": [
    {
      "source": "data/Guide to understanding CPF LIFE.pdf",
      "page": 3,
      "snippet": "The Standard Plan provides..."
    }
  ]
}
```

`page` is the PDF page number (0-indexed) for PDF sources, or the CSV row number for CSV sources.

---

## Configuration

All settings live in `.env` at the project root. The file is git-ignored (never committed).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Model used for answer generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model used for embedding |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where Chroma stores its SQLite DB |
| `DATA_DIR` | `./data` | Directory scanned for corpus files |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `RETRIEVER_K` | `4` | Number of chunks retrieved per query |

To switch to a different Ollama model (e.g. `mistral` or `phi3`), update `OLLAMA_LLM_MODEL` and
restart the server. The embedding model should generally stay as `nomic-embed-text` unless you
re-ingest with the new model.

---

## Setup & Running

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### First-time setup

```bash
# 1. Clone the repo
git clone https://github.com/BenAu001/<repo-name>.git
cd <repo-name>

# 2. Pull required Ollama models (one-time, ~2.3 GB total)
ollama pull llama3.2
ollama pull nomic-embed-text

# 3. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Add your corpus files
cp /path/to/your/files/*.pdf data/
cp /path/to/your/file.csv data/

# 5. Start the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Or use the convenience script:

```bash
./start.sh
```

### Usage

1. Open `http://localhost:8000` in a browser.
2. **Ingest tab** — click **Ingest Documents**. Wait for the success message showing chunk count.
3. **Ask tab** — type a question and press **Ask** or `Ctrl+Enter`.
4. Expand the source cards below the answer to read the retrieved passages.

### Subsequent sessions

```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The Chroma database persists between sessions — no need to re-ingest unless the corpus changes.

---

## How It Works — Step by Step

### Ingest pipeline (`backend/ingest.py`)

1. **File discovery** — `list_data_files()` scans `data/` for `.pdf` and `.csv` files.
2. **Loading**
   - PDFs → `PyPDFLoader` (via `pypdf`). Each page becomes a `Document` with `source` and `page` metadata.
   - CSVs → `CSVLoader`. Each row becomes a `Document` with `source` and `row` metadata.
3. **Splitting** — `RecursiveCharacterTextSplitter` splits documents into chunks of up to 1000
   characters with 200-character overlap. Overlap preserves context across chunk boundaries.
4. **Embedding** — `OllamaEmbeddings(model="nomic-embed-text")` converts each chunk's text into a
   768-dimensional dense vector by calling the local Ollama API.
5. **Storage** — A `chromadb.PersistentClient` opens (or creates) the SQLite-backed store at
   `chroma_db/`. The old `rag_corpus` collection is deleted if present, then `Chroma.from_documents()`
   writes all vectors and their metadata in one batch.

### Query pipeline (`backend/rag.py`)

1. **Retrieval** — The user's question is embedded with `nomic-embed-text`. Chroma performs a
   cosine similarity search and returns the top-4 chunks (`RETRIEVER_K=4`).
2. **Prompt assembly** — A `ChatPromptTemplate` injects the retrieved chunks as `{context}` and
   the question as `{question}` into a strict grounding prompt that instructs the model to answer
   only from the provided context.
3. **Generation** — `ChatOllama(model="llama3.2")` calls the local Ollama API and streams the
   completion through `StrOutputParser` to produce a plain string answer.
4. **Source attribution** — The same top-4 docs retrieved in step 1 are returned alongside the
   answer, each with file name, page/row number, and a 300-character snippet.

### LCEL chain structure

```python
chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | ChatPromptTemplate
    | ChatOllama
    | StrOutputParser
)
```

This is LangChain Expression Language (LCEL). It composes the retriever, prompt, LLM, and parser
into a single callable, making the data flow explicit and easy to modify.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Single shared `chromadb.PersistentClient`** | Avoids SQLite "readonly database" errors that occur when multiple `PersistentClient` instances open the same file concurrently (e.g. a status check racing with an ingest). |
| **Delete-then-recreate collection on ingest** | Guarantees a clean rebuild — no stale chunks from old files accumulate. Simple and predictable. |
| **`nomic-embed-text` for embeddings** | Best-in-class local embedding model available via Ollama. 768-dimensional vectors with strong semantic retrieval quality. |
| **`llama3.2` as the LLM** | Compact (2 GB) yet capable model available locally. Easy to swap via `.env`. |
| **Single `index.html` frontend** | Zero build tooling. The UI is served directly by FastAPI's `StaticFiles`. No npm, no webpack. |
| **Strict grounding prompt** | The prompt explicitly restricts the LLM to the provided context and instructs it to admit when it doesn't know. Reduces hallucination risk. |
| **Chunk overlap of 200** | Sentences that span a chunk boundary appear in both chunks, so retrieval doesn't lose a fact just because it straddles a split point. |

---

## Known Limitations

- **Re-ingest required on corpus change** — adding or removing files from `data/` does not
  automatically update the vector store. Click Ingest Documents again.
- **No streaming** — the `/api/ask` response waits for the full LLM generation before returning.
  For long answers this can feel slow. Streaming via SSE is a natural next step.
- **Single-turn only** — the RAG chain does not maintain conversation history. Each question is
  answered independently.
- **CSV row as a unit** — the CSV loader treats each row as one document. For wide tables with many
  columns, rows can be long and may not split meaningfully.
- **Local only** — designed for `localhost`. Exposing to a network requires adding authentication.
