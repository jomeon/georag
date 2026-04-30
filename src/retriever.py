import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Klasa zarządzająca bazą wektorową Chroma oraz generowaniem embeddingów.
    Pozwala na budowanie nowej bazy z dokumentów oraz ładowanie istniejącej.
    """

    def __init__(self, persist_directory: str = "data/processed/chroma_db"):
        """
        Inicjalizuje menedżera bazy wektorowej.
        
        Args:
            persist_directory (str): Ścieżka do folderu, w którym baza zostanie zapisana na dysku.
        """
        self.persist_directory = persist_directory
        
       
        logger.info("Inicjalizacja modelu embeddingowego (może to zająć chwilę przy pierwszym uruchomieniu)...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.db: Optional[Chroma] = None

    def build_database(self, chunked_documents: List[Document]) -> Chroma:
        """
        Tworzy bazę wektorową z przekazanych fragmentów tekstu, 
        generuje dla nich wektory i zapisuje trwale na dysku.
        
        Args:
            chunked_documents (List[Document]): Lista podzielonych dokumentów z metadanymi.
            
        Returns:
            Chroma: Instancja bazy wektorowej gotowa do zapytań.
        """
        logger.info(f"Rozpoczynam wektoryzację {len(chunked_documents)} fragmentów i zapis do bazy Chroma...")
       
        self.db = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        logger.info(f"Baza wektorowa została pomyślnie zbudowana i zapisana w {self.persist_directory}.")
        return self.db

    def load_database(self) -> Chroma:
        """
        Ładuje wcześniej wygenerowaną bazę wektorową z dysku.
        Zabezpiecza przed koniecznością ponownego przeliczania wektorów.
        
        Returns:
            Chroma: Załadowana instancja bazy wektorowej.
        """
        if not os.path.exists(self.persist_directory):
            logger.error("Baza wektorowa nie istnieje. Najpierw użyj metody build_database.")
            raise FileNotFoundError(f"Katalog {self.persist_directory} nie istnieje.")
            
        logger.info(f"Ładowanie bazy wektorowej z katalogu: {self.persist_directory}...")
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        logger.info("Baza wektorowa załadowana pomyślnie.")
        return self.db