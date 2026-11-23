"""Evaluate the RAG graph on a labelled question set.

Metrics
-------
* retrieval_hit_rate   - answerable questions where an expected source was retrieved
* citation_precision   - cited chunks that come from an expected source
* refusal_accuracy     - unanswerable questions correctly answered with "I don't know"
* false_refusal_rate   - answerable questions wrongly refused
* judge_correctness    - LLM-as-judge score (1-5) against the reference answer, scaled to 0-1
* p50 / p95 latency

    python -m evals.run_eval                      # uses RAG_* settings (local Llama via Ollama)
    python -m evals.run_eval --limit 3            # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

from agentic_rag.core import get_chat_model, get_embeddings, get_settings
from agentic_rag.graph import REFUSAL, ask, build_graph
from agentic_rag.ingest import load_or_build_index

HERE = Path(__file__).parent

JUDGE_PROMPT = """You are grading an AI assistant's answer against a reference answer.
Question: {question}
Reference answer: {reference}
Assistant answer: {answer}

Score factual correctness from 1 (wrong or missing key facts) to 5 (fully correct and complete).
Reply with only the number."""


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_file(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def judge_score(judge, question: str, reference: str, answer: str) -> float | None:
    reply = judge.invoke(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer))
    match = re.search(r"[1-5]", getattr(reply, "content", str(reply)))
    return (int(match.group()) - 1) / 4 if match else None


def score(results: list[dict], dataset: list[dict]) -> dict:
    by_id = {row["id"]: row for row in dataset}
    answerable = [r for r in results if by_id[r["id"]]["answerable"]]
    unanswerable = [r for r in results if not by_id[r["id"]]["answerable"]]

    def hit(r):
        expected = set(by_id[r["id"]]["expected_sources"])
        return any(_source_file(s) in expected for s in r["retrieved_sources"])

    cited = [(c, set(by_id[r["id"]]["expected_sources"])) for r in answerable for c in r["citations"]]
    judged = [r["judge"] for r in answerable if r.get("judge") is not None]
    latencies = sorted(r["latency_ms"] for r in results)

    def ratio(numerator, denominator):
        return round(numerator / denominator, 3) if denominator else None

    return {
        "questions": len(results),
        "retrieval_hit_rate": ratio(sum(hit(r) for r in answerable), len(answerable)),
        "citation_precision": ratio(sum(_source_file(c) in exp for c, exp in cited), len(cited)),
        "refusal_accuracy": ratio(sum(r["answer"] == REFUSAL for r in unanswerable), len(unanswerable)),
        "false_refusal_rate": ratio(sum(r["answer"] == REFUSAL for r in answerable), len(answerable)),
        "judge_correctness": round(statistics.mean(judged), 3) if judged else None,
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else None,
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
    }


def write_report(metrics: dict, results: list[dict], model: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    detail = "\n".join(
        f"| {r['id']} | {'refused' if r['answer'] == REFUSAL else 'answered'} | {', '.join(r['citations']) or '-'} "
        f"| {r.get('judge', '-')} | {r['latency_ms']} |"
        for r in results
    )
    report = (
        f"# Evaluation report\n\nModel: `{model}` - run {stamp}\n\n| Metric | Value |\n|---|---|\n{rows}\n\n"
        f"## Per question\n\n| id | outcome | citations | judge (0-1) | latency ms |\n|---|---|---|---|---|\n{detail}\n"
    )
    path = out_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps({"metrics": metrics, "results": results}, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.jsonl")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    store = load_or_build_index(settings, get_embeddings(settings))
    app = build_graph(get_chat_model(settings), store, settings)
    judge = get_chat_model(settings, settings.judge_model)

    dataset = load_dataset(args.dataset)[: args.limit]
    results = []
    for row in dataset:
        result = ask(app, row["question"])
        result["id"] = row["id"]
        if row["answerable"] and result["answer"] != REFUSAL:
            result["judge"] = judge_score(judge, row["question"], row["reference"], result["answer"])
        results.append(result)
        outcome = "REFUSED " if result["answer"] == REFUSAL else "answered"
        print(f"{row['id']:<18} {outcome} {result['latency_ms']:>8} ms")

    metrics = score(results, dataset)
    path = write_report(metrics, results, settings.llm_model, args.out)
    print(json.dumps(metrics, indent=2))
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
