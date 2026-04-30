import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Klasa odpowiedzialna za dzielenie tekstu (obiektów Document) 
    na mniejsze fragmenty (chunki) zachowujące sens semantyczny.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicjalizuje chunker z zadanymi parametrami.
        
        Args:
            chunk_size (int): Maksymalna liczba znaków w jednym fragmencie.
            chunk_overlap (int): Liczba znaków, które nakładają się między sąsiadującymi fragmentami.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Wykorzystujemy RecursiveCharacterTextSplitter, który stara się
        # nie rozrywać akapitów i zdań w połowie.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Dzieli listę wczytanych dokumentów na mniejsze chunki.
        
        Args:
            documents (List[Document]): Lista wczytanych stron dokumentów.
            
        Returns:
            List[Document]: Lista podzielonych fragmentów (chunków) gotowych do wektoryzacji.
        """
        logger.info(f"Rozpoczynam podział {len(documents)} stron na mniejsze fragmenty...")
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Podzielono tekst na {len(chunked_docs)} chunków.")
        
        return chunked_docs