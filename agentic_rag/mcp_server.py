"""MCP server: lets Claude Desktop, Claude Code or any MCP client query your documents.

python -m agentic_rag.mcp_server
"""

from __future__ import annotations

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from agentic_rag.core import get_chat_model, get_embeddings, get_settings
from agentic_rag.graph import ask, build_graph
from agentic_rag.ingest import load_or_build_index

server = MCPServer(
    name="llama-agentic-rag",
    instructions="Answers questions from the indexed company documents, with citations. "
    "Returns 'I don't know' when the documents do not support an answer.",
)


@lru_cache
def _rag():
    settings = get_settings()
    store = load_or_build_index(settings, get_embeddings(settings))
    return build_graph(get_chat_model(settings), store, settings)


@server.tool()
def ask_documents(question: str) -> dict:
    """Answer a question from the indexed documents. Returns the answer, cited chunk ids and a groundedness flag."""
    result = ask(_rag(), question)
    return {key: result[key] for key in ("answer", "citations", "grounded")}


if __name__ == "__main__":
    server.run()
