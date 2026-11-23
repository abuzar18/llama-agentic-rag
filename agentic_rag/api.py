"""FastAPI service exposing the RAG graph.

uvicorn agentic_rag.api:app --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentic_rag.core import get_chat_model, get_embeddings, get_settings
from agentic_rag.graph import ask, build_graph
from agentic_rag.ingest import build_index, load_or_build_index

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    grounded: bool
    retrieved_sources: list[str]
    steps: list[str]
    latency_ms: float


def create_app(rag=None) -> FastAPI:
    """Create the API. Pass a compiled graph as ``rag`` to skip model loading (used in tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if rag is not None:
            app.state.rag = rag
        else:
            settings = get_settings()
            store = load_or_build_index(settings, get_embeddings(settings))
            app.state.rag = build_graph(get_chat_model(settings), store, settings)
            logger.info("RAG graph ready (model=%s)", settings.llm_model)
        yield

    app = FastAPI(title="Llama Agentic RAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask_endpoint(request: AskRequest) -> dict:
        try:
            return ask(app.state.rag, request.question)
        except Exception as exc:  # surface model/back-end failures as 503, not 500 stack traces
            logger.exception("ask failed")
            raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {type(exc).__name__}") from exc

    @app.post("/ingest")
    def ingest_endpoint() -> dict:
        settings = get_settings()
        store = build_index(settings, get_embeddings(settings))
        app.state.rag = build_graph(get_chat_model(settings), store, settings)
        return {"status": "reindexed"}

    return app


app = create_app()
