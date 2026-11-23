"""Load documents, split them into chunks and build a persisted vector index.

Run directly to (re)build the index:

    python -m agentic_rag.ingest
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.core import Settings, get_embeddings, get_settings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(docs_dir: str | Path) -> list[Document]:
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {root}")
    docs = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": path.relative_to(root).as_posix()}))
    if not docs:
        raise ValueError(f"No .md or .txt documents found in {root}")
    return docs


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}#{counters[source]}"
    return chunks


def build_index(settings: Settings, embeddings: Embeddings) -> InMemoryVectorStore:
    chunks = split_documents(load_documents(settings.docs_dir), settings.chunk_size, settings.chunk_overlap)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    index_path = Path(settings.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(index_path))
    logger.info("Indexed %d chunks from %s into %s", len(chunks), settings.docs_dir, index_path)
    return store


def load_or_build_index(settings: Settings, embeddings: Embeddings) -> InMemoryVectorStore:
    index_path = Path(settings.index_path)
    if index_path.exists():
        return InMemoryVectorStore.load(str(index_path), embeddings)
    return build_index(settings, embeddings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = get_settings()
    build_index(cfg, get_embeddings(cfg))
