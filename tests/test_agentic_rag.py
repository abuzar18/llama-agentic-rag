"""Offline tests: deterministic fake embeddings and scripted fake LLM replies, no GPU or network needed."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agentic_rag.api import create_app
from agentic_rag.core import Settings
from agentic_rag.graph import REFUSAL, ask, build_graph, extract_citations, is_yes
from agentic_rag.ingest import build_index, load_documents, split_documents
from evals.run_eval import score

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(provider="fake", docs_dir=str(DOCS), index_path=str(tmp_path / "store.json"), top_k=2)


@pytest.fixture
def store(settings):
    return build_index(settings, DeterministicFakeEmbedding(size=64))


def run(store, settings, replies: list[str], question: str = "How many days of annual leave do I get?") -> dict:
    llm = FakeListChatModel(responses=replies)
    return ask(build_graph(llm, store, settings), question)


def test_ingest_adds_chunk_ids_and_persists(settings, store):
    chunks = split_documents(load_documents(DOCS), settings.chunk_size, settings.chunk_overlap)
    assert chunks and all("#" in c.metadata["chunk_id"] for c in chunks)
    assert Path(settings.index_path).exists()


def test_happy_path_returns_grounded_cited_answer(store, settings):
    # grade doc1, grade doc2, generate, groundedness check
    result = run(store, settings, ["yes", "no", "Employees get 24 days of paid leave [1].", "yes"])
    assert result["grounded"] is True
    assert result["answer"].startswith("Employees get 24 days")
    assert len(result["citations"]) == 1
    assert result["steps"] == ["retrieve(2)", "grade(1/2)", "generate", "grounded(yes)"]


def test_rewrites_once_then_refuses_when_nothing_relevant(store, settings):
    result = run(store, settings, ["no", "no", "annual leave policy days", "no", "no"])
    assert result["answer"] == REFUSAL
    assert result["steps"] == ["retrieve(2)", "grade(0/2)", "rewrite", "retrieve(2)", "grade(0/2)", "refuse"]


def test_ungrounded_answer_is_replaced_with_refusal(store, settings):
    result = run(store, settings, ["yes", "yes", "You get 40 days of leave [1].", "no"])
    assert result["answer"] == REFUSAL
    assert result["grounded"] is False
    assert result["citations"] == []


def test_answer_without_citations_is_rejected(store, settings):
    result = run(store, settings, ["yes", "yes", "You get 24 days of leave.", "yes"])
    assert result["answer"] == REFUSAL


def test_reasoning_tags_are_stripped(store, settings):
    replies = ["<think>hmm</think>yes", "no", "<think>x</think>24 days [1].", "<think>ok</think> Yes."]
    result = run(store, settings, replies)
    assert result["answer"] == "24 days [1]."
    assert result["grounded"] is True


@pytest.mark.parametrize(
    "text,expected", [("yes", True), ("Yes.", True), ("**yes**", True), ("no", False), ("", False)]
)
def test_is_yes(text, expected):
    assert is_yes(text) is expected


def test_extract_citations_ignores_out_of_range(store):
    docs = store.similarity_search("leave", k=2)
    assert extract_citations("A [1]. B [1]. C [9].", docs) == [docs[0].metadata["chunk_id"]]


def test_api_ask_and_health(store, settings):
    llm = FakeListChatModel(responses=["yes", "no", "24 days [1].", "yes"])
    with TestClient(create_app(rag=build_graph(llm, store, settings))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        body = client.post("/ask", json={"question": "How much annual leave?"}).json()
        assert body["grounded"] is True and body["citations"]
        assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_eval_scoring():
    dataset = [
        {"id": "a", "answerable": True, "expected_sources": ["employee-handbook.md"]},
        {"id": "b", "answerable": False, "expected_sources": []},
    ]
    results = [
        {
            "id": "a",
            "answer": "24 days [1].",
            "citations": ["employee-handbook.md#2"],
            "retrieved_sources": ["employee-handbook.md#2"],
            "latency_ms": 10.0,
            "judge": 1.0,
        },
        {"id": "b", "answer": REFUSAL, "citations": [], "retrieved_sources": [], "latency_ms": 20.0},
    ]
    metrics = score(results, dataset)
    assert metrics["retrieval_hit_rate"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
