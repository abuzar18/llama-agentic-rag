"""Settings and model factories.

Everything is configured through environment variables prefixed with ``RAG_``
(or a local ``.env`` file), so the same code runs on a laptop GPU, in Docker
and in CI without edits.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # "ollama" runs Llama locally; "fake" swaps in deterministic test doubles.
    provider: Literal["ollama", "fake"] = "ollama"
    llm_model: str = "llama3.1:8b"
    judge_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    docs_dir: str = "data/docs"
    index_path: str = ".index/store.json"
    chunk_size: int = 800
    chunk_overlap: int = 120

    top_k: int = 4
    min_relevant_docs: int = 1
    max_rewrites: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_chat_model(settings: Settings, model: str | None = None) -> BaseChatModel:
    if settings.provider == "fake":
        raise ValueError("The 'fake' provider has no chat model; inject one explicitly in tests.")
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.provider == "fake":
        return DeterministicFakeEmbedding(size=256)
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
