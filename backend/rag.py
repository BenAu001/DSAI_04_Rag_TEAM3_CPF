import os
import time
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.ingest import get_vectorstore

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "6"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.0"))

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the following question as a short, keyword-rich search query optimised \
for semantic retrieval over a document corpus. \
Keep the core meaning; expand pronouns and abbreviations into explicit terms. \
Output ONLY the rewritten query, no explanation.

Question: {question}
Search query:"""
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. \
Answer the question accurately and thoroughly using ONLY the provided context.

Guidelines:
- Be specific — include names, actions, quotes, and details present in the context
- Use bullet points or numbered lists when listing multiple items or steps
- If the context does not contain enough information to answer fully, say so explicitly
- Do NOT add information, opinions, or disclaimers beyond what the context supports

Context:
{context}

Question: {question}

Answer:"""
)


def _filename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]


def ask(question: str) -> dict:
    t_start = time.perf_counter()
    trace = []

    def elapsed_ms():
        return round((time.perf_counter() - t_start) * 1000)

    # ── Step 1: record original query ────────────────────────────
    trace.append({
        "step": 1, "type": "query", "label": "Original Query",
        "ts_ms": 0, "duration_ms": 0,
        "data": {"text": question},
    })

    # ── Step 2: query rewrite ────────────────────────────────────
    llm_fast = ChatOllama(
        model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0, num_ctx=512,
    )
    t0 = time.perf_counter()
    try:
        rewritten = (REWRITE_PROMPT | llm_fast | StrOutputParser()).invoke({"question": question}).strip()
        # Strip any leading label the model may have added
        for prefix in ("Search query:", "Query:", "Rewritten:"):
            if rewritten.lower().startswith(prefix.lower()):
                rewritten = rewritten[len(prefix):].strip()
    except Exception:
        rewritten = question
    rewrite_ms = round((time.perf_counter() - t0) * 1000)

    trace.append({
        "step": 2, "type": "rewrite", "label": "Query Rewrite",
        "ts_ms": elapsed_ms() - rewrite_ms, "duration_ms": rewrite_ms,
        "data": {"original": question, "rewritten": rewritten},
    })

    # ── Step 3: vector retrieval ─────────────────────────────────
    vectorstore = get_vectorstore()
    t0 = time.perf_counter()
    scored_docs = vectorstore.similarity_search_with_relevance_scores(rewritten, k=RETRIEVER_K)
    retrieve_ms = round((time.perf_counter() - t0) * 1000)

    trace.append({
        "step": 3, "type": "retrieve", "label": "Vector Retrieval",
        "ts_ms": elapsed_ms() - retrieve_ms, "duration_ms": retrieve_ms,
        "data": {
            "query_used": rewritten,
            "k": RETRIEVER_K,
            "results": [
                {
                    "rank": i + 1,
                    "source": _filename(doc.metadata.get("source", "unknown")),
                    "page": doc.metadata.get("page", doc.metadata.get("row", None)),
                    "score": round(float(score), 4),
                }
                for i, (doc, score) in enumerate(scored_docs)
            ],
        },
    })

    # ── Step 4: rerank (filter low-score, sort desc) ─────────────
    t0 = time.perf_counter()
    reranked = [(doc, score) for doc, score in scored_docs if score >= RERANK_MIN_SCORE]
    if not reranked:
        reranked = list(scored_docs)
    reranked.sort(key=lambda x: x[1], reverse=True)
    rerank_ms = round((time.perf_counter() - t0) * 1000)

    scores = [s for _, s in reranked]
    trace.append({
        "step": 4, "type": "rerank", "label": "Score Rerank",
        "ts_ms": elapsed_ms() - rerank_ms, "duration_ms": rerank_ms,
        "data": {
            "kept": len(reranked),
            "filtered_out": len(scored_docs) - len(reranked),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "avg_score": round(sum(scores) / len(scores), 4),
        },
    })

    # ── Step 5: LLM generation ───────────────────────────────────
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in reranked)

    llm_gen = ChatOllama(
        model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL,
        temperature=0, num_ctx=4096,
    )
    gen_chain = RAG_PROMPT | llm_gen | StrOutputParser()

    t0 = time.perf_counter()
    answer = gen_chain.invoke({"context": context, "question": question})
    llm_ms = round((time.perf_counter() - t0) * 1000)

    trace.append({
        "step": 5, "type": "generate", "label": "LLM Generation",
        "ts_ms": elapsed_ms() - llm_ms, "duration_ms": llm_ms,
        "data": {
            "model": OLLAMA_LLM_MODEL,
            "context_chunks": len(reranked),
            "answer_chars": len(answer),
        },
    })

    total_ms = elapsed_ms()

    sources = []
    for rank, (doc, score) in enumerate(reranked, start=1):
        meta = doc.metadata
        sources.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", meta.get("row", None)),
            "snippet": doc.page_content[:300],
            "score": round(float(score), 4),
            "rank": rank,
        })

    return {
        "answer": answer,
        "sources": sources,
        "trace": trace,
        "metrics": {
            "llm_ms": llm_ms,
            "retrieve_ms": retrieve_ms,
            "rewrite_ms": rewrite_ms,
            "total_ms": total_ms,
            "chunks_retrieved": len(scored_docs),
            "chunks_used": len(reranked),
            "llm_model": OLLAMA_LLM_MODEL,
        },
    }
