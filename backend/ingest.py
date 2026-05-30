import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
COLLECTION_NAME = "rag_corpus"

SUPPORTED_EXTENSIONS = {".pdf", ".csv"}


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _get_chroma_client() -> chromadb.PersistentClient:
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def list_data_files() -> list[dict]:
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        return []
    files = []
    for f in sorted(data_path.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append({"name": f.name, "size": f.stat().st_size, "type": f.suffix.lstrip(".")})
    return files


def load_documents(file_path: Path) -> list:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    elif suffix == ".csv":
        loader = CSVLoader(str(file_path))
        return loader.load()
    return []


def ingest_all() -> dict:
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        return {"status": "error", "message": f"Data directory '{DATA_DIR}' not found."}

    files = list_data_files()
    if not files:
        return {"status": "error", "message": "No supported files found in the data directory."}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_docs = []
    processed_files = []
    errors = []

    for file_info in files:
        file_path = data_path / file_info["name"]
        try:
            raw_docs = load_documents(file_path)
            chunks = splitter.split_documents(raw_docs)
            all_docs.extend(chunks)
            processed_files.append(file_info["name"])
        except Exception as e:
            errors.append({"file": file_info["name"], "error": str(e)})

    if not all_docs:
        return {"status": "error", "message": "No documents could be loaded.", "errors": errors}

    embeddings = get_embeddings()

    # Use a single client — delete old collection if it exists, then recreate
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client,
    )

    return {
        "status": "success",
        "documents_ingested": len(all_docs),
        "files_processed": processed_files,
        "errors": errors,
    }


def get_vectorstore() -> Chroma:
    client = _get_chroma_client()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        client=client,
    )


def get_document_count() -> int:
    try:
        client = _get_chroma_client()
        col = client.get_collection(COLLECTION_NAME)
        return col.count()
    except Exception:
        return 0
