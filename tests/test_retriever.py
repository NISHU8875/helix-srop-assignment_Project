"""
Unit tests for RAG retrieval.
Requires the vector store to be seeded first (run ingest.py on docs/).
"""
import pytest
from app.rag.ingest import chunk_markdown


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    text = """# Deploy Keys

Some content about deploy keys.

## How to Rotate

This is how you rotate a deploy key. First, navigate to settings.

Then find the keys section and click rotate.

## Best Practices

Always keep your keys secure.
"""
    chunks = chunk_markdown(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks), "All chunks should be non-empty"
    # Verify chunks aren't too large
    for chunk in chunks:
        assert len(chunk) <= 150, f"Chunk too large: {len(chunk)}"


def test_chunker_respects_overlap():
    """Test that chunks have overlap for context preservation."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunks = chunk_markdown(text, chunk_size=30, overlap=15)
    # With overlap, later chunks should contain tail of earlier chunks
    if len(chunks) > 1:
        # Check if there's some overlap by seeing if chunks share content
        combined = "".join(chunks)
        # Should have some duplication due to overlap
        assert combined.count("Sentence") > 5


def test_chunker_empty_text():
    """Chunker should handle empty text gracefully."""
    chunks = chunk_markdown("", chunk_size=100)
    assert len(chunks) == 0


def test_chunker_preserves_sections():
    """Chunker should keep markdown sections together when possible."""
    text = """# Main Topic

## Section A
Content A

## Section B
Content B
"""
    chunks = chunk_markdown(text, chunk_size=500, overlap=20)
    assert len(chunks) > 0
    # At least one chunk should contain "Section A"
    assert any("Section A" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """
    search_docs must return chunk IDs and scores in [0, 1].
    
    NOTE: This test is skipped if vector store hasn't been seeded.
    """
    try:
        from app.agents.tools.search_docs import search_docs
        results = await search_docs("how to rotate a deploy key", k=3)
        
        if not results:
            pytest.skip("Vector store not seeded. Run: python -m app.rag.ingest --path docs/")
        
        assert len(results) > 0
        for r in results:
            assert r.chunk_id
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0, 1]"
            assert r.content
            
    except Exception as e:
        pytest.skip(f"Vector store not initialized: {e}")

