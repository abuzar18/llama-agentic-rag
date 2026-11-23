"""Self-correcting, citation-grounded RAG workflow built with LangGraph.

Flow::

    retrieve -> grade --(enough relevant docs)--> generate -> check_grounded -> END
                  |                                              |
                  +--(none relevant, rewrites left)--> rewrite --+--> retrieve
                  +--(none relevant, no rewrites left)--> refuse -> END

Design choices:
* Every chunk is graded for relevance before it reaches the generator, so
  off-topic context cannot leak into the answer.
* The generator must cite sources as [n]; citations are mapped back to
  document chunk ids and returned to the caller.
* A separate groundedness check rejects answers that are not supported by the
  cited context. The system prefers "I don't know" over a confident guess.
* Prompts ask for one-word verdicts and parsing is tolerant, which keeps the
  graph reliable with small local models such as Llama 3.1 8B.
"""

from __future__ import annotations

import re
import time
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from agentic_rag.core import Settings

REFUSAL = "I don't know based on the provided documents."

GRADE_PROMPT = """You are grading whether a document chunk helps answer a question.
Question: {question}
Chunk:
\"\"\"{chunk}\"\"\"
Reply with exactly one word: yes or no."""

REWRITE_PROMPT = """The search query below returned no relevant documents.
Rewrite it as a short, specific search query that uses likely keywords from an
internal company handbook. Return only the rewritten query.
Original question: {question}"""

GENERATE_PROMPT = """Answer the question using ONLY the numbered context below.
Rules:
- Cite every factual sentence with its source number in square brackets, e.g. [1].
- If the context does not contain the answer, reply exactly: {refusal}
- Be concise: at most 4 sentences.

Context:
{context}

Question: {question}
Answer:"""

GROUNDED_PROMPT = """You are a strict fact checker.
Context:
{context}

Answer:
{answer}

Is every claim in the answer directly supported by the context?
Reply with exactly one word: yes or no."""


class RAGState(TypedDict, total=False):
    question: str
    query: str
    documents: list[Document]
    retrieved_sources: list[str]
    rewrites: int
    answer: str
    citations: list[str]
    grounded: bool
    steps: list[str]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _text(message) -> str:
    """Return model output as plain text, dropping any <think> reasoning blocks."""
    content = message.content if hasattr(message, "content") else str(message)
    if isinstance(content, list):
        content = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return _THINK_RE.sub("", content).strip()


def is_yes(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and words[0] == "yes"


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {doc.metadata.get('chunk_id', doc.metadata.get('source', 'unknown'))})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def extract_citations(answer: str, docs: list[Document]) -> list[str]:
    cited = []
    for number in _CITATION_RE.findall(answer):
        index = int(number) - 1
        if 0 <= index < len(docs):
            chunk_id = docs[index].metadata.get("chunk_id", docs[index].metadata.get("source", "unknown"))
            if chunk_id not in cited:
                cited.append(chunk_id)
    return cited


def build_graph(llm: BaseChatModel, store: VectorStore, settings: Settings, judge: BaseChatModel | None = None):
    judge = judge or llm

    def retrieve(state: RAGState) -> RAGState:
        query = state.get("query") or state["question"]
        docs = store.similarity_search(query, k=settings.top_k)
        return {
            "query": query,
            "documents": docs,
            "retrieved_sources": [d.metadata.get("chunk_id", d.metadata.get("source", "")) for d in docs],
            "steps": [*state.get("steps", []), f"retrieve({len(docs)})"],
        }

    def grade(state: RAGState) -> RAGState:
        relevant = []
        for doc in state["documents"]:
            verdict = _text(llm.invoke(GRADE_PROMPT.format(question=state["question"], chunk=doc.page_content)))
            if is_yes(verdict):
                relevant.append(doc)
        return {"documents": relevant, "steps": [*state["steps"], f"grade({len(relevant)}/{settings.top_k})"]}

    def route_after_grade(state: RAGState) -> str:
        if len(state["documents"]) >= settings.min_relevant_docs:
            return "generate"
        if state.get("rewrites", 0) < settings.max_rewrites:
            return "rewrite"
        return "refuse"

    def rewrite(state: RAGState) -> RAGState:
        query = _text(llm.invoke(REWRITE_PROMPT.format(question=state["question"]))).splitlines()[0].strip('" ')
        return {
            "query": query or state["question"],
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": [*state["steps"], "rewrite"],
        }

    def generate(state: RAGState) -> RAGState:
        docs = state["documents"]
        prompt = GENERATE_PROMPT.format(context=format_context(docs), question=state["question"], refusal=REFUSAL)
        answer = _text(llm.invoke(prompt))
        return {"answer": answer, "citations": extract_citations(answer, docs), "steps": [*state["steps"], "generate"]}

    def check_grounded(state: RAGState) -> RAGState:
        if state["answer"].strip() == REFUSAL:
            return {"grounded": True, "citations": [], "steps": [*state["steps"], "grounded(refusal)"]}
        verdict = _text(
            judge.invoke(GROUNDED_PROMPT.format(context=format_context(state["documents"]), answer=state["answer"]))
        )
        if is_yes(verdict) and state["citations"]:
            return {"grounded": True, "steps": [*state["steps"], "grounded(yes)"]}
        return {"answer": REFUSAL, "citations": [], "grounded": False, "steps": [*state["steps"], "grounded(no)"]}

    def refuse(state: RAGState) -> RAGState:
        return {"answer": REFUSAL, "citations": [], "grounded": True, "steps": [*state["steps"], "refuse"]}

    graph = StateGraph(RAGState)
    for name, node in [
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("generate", generate),
        ("check_grounded", check_grounded),
        ("refuse", refuse),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, ["generate", "rewrite", "refuse"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "check_grounded")
    graph.add_edge("check_grounded", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def ask(app, question: str) -> dict:
    """Run the graph and return a JSON-serialisable result with latency."""
    started = time.perf_counter()
    state = app.invoke({"question": question, "steps": [], "rewrites": 0})
    return {
        "question": question,
        "answer": state["answer"],
        "citations": state.get("citations", []),
        "grounded": state.get("grounded", False),
        "retrieved_sources": state.get("retrieved_sources", []),
        "steps": state.get("steps", []),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
