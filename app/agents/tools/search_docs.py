"""
search_docs tool — used by KnowledgeAgent.

Queries the vector store for relevant documentation chunks.
Returns chunk IDs, scores, and content so the agent can cite sources.

TODO for candidate: implement this tool.
Wire it to your chosen vector store (Chroma, LanceDB, FAISS, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
import chromadb


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict  # e.g. {"product_area": "security", "source": "deploy-keys.md"}


# Global Chroma client (initialize once)
_chroma_client: chromadb.PersistentClient | None = None


def _get_chroma_client() -> chromadb.PersistentClient:
    """Get or create Chroma client (singleton pattern)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return _chroma_client


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the vector store for top-k relevant chunks.

    Args:
        query: natural language query from the user
        k: number of chunks to return
        product_area: optional metadata filter (e.g. "security", "ci-cd")

    Returns:
        List of DocChunk ordered by descending similarity score.

    Design considerations:
    - How do you embed the query? Same model as at ingest time.
    - Do you apply a score threshold to filter low-quality results?
    - How do you format chunks for the agent? Include chunk_id so agent can cite.
    """
    client = _get_chroma_client()

    # Get or create collection
    try:
        collection = client.get_collection(name="helix_docs")
    except Exception:
        # Collection doesn't exist yet (ingest not run)
        return []

    # Build where filter if product_area is specified
    where_filter = None
    if product_area:
        where_filter = {"product_area": product_area}

    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    # Convert distances to similarity scores (cosine distance → similarity)
    # Chroma returns distances; convert to [0, 1] similarity
    chunks = []
    for i, (doc, meta, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        # Cosine distance in [0, 2]; convert to similarity in [0, 1]
        similarity_score = 1 - (distance / 2)
        similarity_score = max(0.0, min(1.0, similarity_score))

        # Extract chunk_id from metadata if present; otherwise generate from doc
        chunk_id = meta.get("chunk_index", str(i))
        if "source" in meta:
            chunk_id = f"{meta['source']}_{chunk_id}"

        chunks.append(
            DocChunk(
                chunk_id=chunk_id,
                score=similarity_score,
                content=doc,
                metadata=meta,
            )
        )

    return chunks

