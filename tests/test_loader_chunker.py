import pytest
from langchain_core.documents import Document
from src.chunker import TextChunker

def test_chunker_splits_correctly():
    """
    Test sprawdza, czy TextChunker poprawnie dzieli zbyt długi tekst 
    na mniejsze fragmenty, nie przekraczając zadanego limitu znaków.
    """
  
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    long_text = "A" * 250 
    doc = Document(page_content=long_text, metadata={"source": "symulacja.pdf"})
    
    chunks = chunker.split_documents([doc])
    
    assert len(chunks) > 1, "Chunker powinien podzielić długi tekst na wiele fragmentów."
    
    for chunk in chunks:
        assert len(chunk.page_content) <= 100, f"Chunk jest za długi: {len(chunk.page_content)} znaków."
        assert chunk.metadata["source"] == "symulacja.pdf", "Chunk zgubił metadane źródłowe!"