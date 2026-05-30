import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from backend.ingest import get_vectorstore

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def ask(question: str) -> dict:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatOllama(
        model=OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Retrieve source docs separately for citation
    source_docs = retriever.invoke(question)

    answer = chain.invoke(question)

    sources = []
    for doc in source_docs:
        meta = doc.metadata
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", meta.get("row", None)),
            "snippet": doc.page_content[:300],
        })

    return {"answer": answer, "sources": sources}
