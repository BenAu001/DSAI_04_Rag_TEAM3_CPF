# How to Swap the Data Source

Follow these steps whenever you want to replace the document corpus with a new set of PDFs.

---

## 1. Stop the server

```bash
pkill -f "uvicorn backend.main"
```

---

## 2. Clear the old data files

Remove everything currently in the `data/` folder:

```bash
rm -f data/*
```

> Keep the `data/` folder itself — only delete the files inside it.

---

## 3. Add your new PDF files

Copy your new PDFs into `data/`:

```bash
cp /path/to/your/files/*.pdf data/
```

**Supported format:** `.pdf` only.  
**Text-based PDFs** are read directly.  
**Scanned / image-based PDFs** are processed automatically with OCR (Tesseract at 300 dpi) — no extra steps needed.

---

## 4. Delete the old vector database

The ChromaDB store lives in `chroma_db/`. Wipe it so stale embeddings from the previous corpus are removed:

```bash
rm -rf chroma_db/
```

---

## 5. (Optional) Update tuning parameters

Edit `.env` to adjust chunking or retrieval for your new content:

| Variable | Default | When to change |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Increase for longer, denser documents |
| `CHUNK_OVERLAP` | `200` | Increase if answers seem to cut off mid-thought |
| `RETRIEVER_K` | `4` | Increase to surface more context per query |
| `OLLAMA_LLM_MODEL` | `llama3.2` | Swap to a larger model for better reasoning |

---

## 6. Restart the server

```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 7. Ingest the new documents

**Via the UI** — open `http://localhost:8000`, check the sidebar shows your new files, then click **Ingest Documents**.

**Via the API** — or run directly from the terminal:

```bash
curl -s -X POST http://localhost:8000/api/ingest | python3 -m json.tool
```

A successful response looks like:

```json
{
  "status": "success",
  "documents_ingested": 42,
  "files_processed": ["goldilocks-and-three-bears.pdf"],
  "errors": []
}
```

---

## 8. Verify

Check the sidebar — the corpus status dot should turn **green** with the new chunk count. Ask a test question to confirm answers are drawn from the new source.

```bash
curl -s http://localhost:8000/api/status
# → {"has_documents": true, "document_count": 42, ...}
```

---

## Quick reference (all steps in one block)

```bash
pkill -f "uvicorn backend.main"

rm -f data/*
cp /path/to/new/files/*.pdf data/
rm -rf chroma_db/

source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

curl -s -X POST http://localhost:8000/api/ingest | python3 -m json.tool
```
