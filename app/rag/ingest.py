"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Reads markdown files, chunks them, embeds, and writes to the vector store.

TODO for candidate: implement chunking and embedding logic.
"""
import argparse
import asyncio
import hashlib
from pathlib import Path
from typing import Any
import re

import chromadb


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split markdown text into overlapping chunks.

    Strategy: Split on heading boundaries first (preserve sections),
    then by sentences if a section is too long.
    This keeps related content together while maintaining chunk size limits.

    Design rationale:
    - Heading-aware: preserves semantic boundaries (e.g., "Deploy Keys" section stays coherent)
    - Sentence-aware: doesn't break mid-sentence for better retrieval quality
    - Overlapping: context preservation at chunk boundaries
    """
    # Split by level 2+ headings first to preserve sections
    section_pattern = r"(?=^## )"
    sections = re.split(section_pattern, text, flags=re.MULTILINE)

    chunks: list[str] = []

    for section in sections:
        if not section.strip():
            continue

        # If section is small enough, add as-is (with overlap from previous)
        if len(section) <= chunk_size + overlap:
            if chunks and overlap > 0:
                # Add tail of previous chunk for context
                tail = chunks[-1][-overlap:] if chunks else ""
                chunks.append(tail + section)
            else:
                chunks.append(section)
            continue

        # Otherwise split section by sentences
        sentences = re.split(r"(?<=[.!?])\s+", section)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += (sentence + " ").strip() + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Add overlap from end of previous chunk
                if overlap > 0 and chunks:
                    current_chunk = chunks[-1][-overlap:] + " " + sentence + " "
                else:
                    current_chunk = sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    return [c for c in chunks if c.strip()]


def extract_metadata(file_path: Path, text: str) -> dict[str, Any]:
    """
    Extract metadata from a markdown file's frontmatter.

    Expected frontmatter format:
        ---
        title: Deploy Keys
        product_area: security
        tags: [keys, secrets]
        ---

    Returns a dict suitable for vector store metadata filtering.
    """
    metadata: dict[str, Any] = {
        "source": file_path.name,
        "file_path": str(file_path),
    }

    # Try to parse YAML frontmatter
    frontmatter_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if frontmatter_match:
        frontmatter = frontmatter_match.group(1)
        # Simple parsing (not a full YAML parser)
        for line in frontmatter.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"\'[]')
                metadata[key] = value

    # Extract first H1 as title if not in frontmatter
    if "title" not in metadata:
        h1_match = re.search(r"^# (.+)$", text, re.MULTILINE)
        if h1_match:
            metadata["title"] = h1_match.group(1)

    # Default product_area if not specified
    if "product_area" not in metadata:
        metadata["product_area"] = "general"

    return metadata


def generate_chunk_id(file_path: Path, chunk_index: int, chunk_text: str) -> str:
    """
    Generate stable, deterministic chunk ID.

    Includes file path, index, and hash of content.
    Re-ingesting the same file produces identical IDs (deduplication).
    """
    content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:8]
    file_stem = file_path.stem
    return f"{file_stem}_{chunk_index:04d}_{content_hash}"


async def embed_chunks(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    """
    Embed chunks using Chroma's default embedding function.
    In production, use Chroma's built-in embedding or OpenAI/Cohere.
    """
    # Chroma client handles embeddings automatically; we just provide texts
    # Return dummy embeddings for now — Chroma will embed when we add to collection
    return [[0.0] * 384 for _ in texts]  # Placeholder


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.

    Design considerations:
    - Generate a stable chunk_id (e.g. sha256(file + chunk_index)) for deduplication.
    - Run embeddings in batches to avoid rate limiting.
    - Print progress so the user can see what's happening.
    """
    md_files = sorted(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    # Initialize Chroma client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for file_path in md_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            metadata = extract_metadata(file_path, text)
            chunks = chunk_markdown(text, chunk_size, chunk_overlap)

            # Build documents to upsert
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_id = generate_chunk_id(file_path, chunk_idx, chunk_text)
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)
                chunk_metadatas.append({**metadata, "chunk_index": chunk_idx})

            # Upsert to collection
            if chunk_ids:
                collection.upsert(
                    ids=chunk_ids,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas,
                )
                total_chunks += len(chunks)
                print(f"  {file_path.name}: {len(chunks)} chunks (IDs: {chunk_ids[:2]}...)")

        except Exception as e:
            print(f"  ERROR in {file_path}: {e}")
            raise

    print(f"\nIngest complete. Total chunks: {total_chunks}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
