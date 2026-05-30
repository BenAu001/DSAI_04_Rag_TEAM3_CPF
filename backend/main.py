import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.ingest import ingest_all, list_data_files, get_document_count
from backend.rag import ask

load_dotenv()

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")


@app.get("/api/status")
def status():
    count = get_document_count()
    return {"has_documents": count > 0, "document_count": count}


@app.get("/api/files")
def list_files():
    return {"files": list_data_files()}


@app.post("/api/ingest")
def ingest():
    try:
        result = ingest_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


class AskRequest(BaseModel):
    question: str


@app.post("/api/ask")
def ask_question(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    doc_count = get_document_count()
    if doc_count == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Run ingest first.")
    try:
        result = ask(body.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
