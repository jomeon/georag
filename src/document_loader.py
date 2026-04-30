import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFDocumentLoader:
    """
    Klasa odpowiedzialna za wczytywanie dokumentów PDF z podanego katalogu
    lub pojedynczego pliku i konwertowanie ich na obiekty LangChain Document.
    """

    def __init__(self, data_path: str):
        """
        Inicjalizuje loader ścieżką do danych.
        
        Args:
            data_path (str): Ścieżka do folderu z PDF-ami lub do pojedynczego pliku PDF.
        """
        self.data_path = data_path

    def load_documents(self) -> List[Document]:
        """
        Wczytuje wszystkie pliki PDF z podanej ścieżki.
        
        Returns:
            List[Document]: Lista załadowanych dokumentów z metadanymi (np. numerem strony).
        """
        documents = []
        
        if os.path.isfile(self.data_path) and self.data_path.lower().endswith('.pdf'):
            logger.info(f"Wczytywanie pojedynczego pliku: {self.data_path}")
            loader = PyPDFLoader(self.data_path)
            documents.extend(loader.load())
        elif os.path.isdir(self.data_path):
            logger.info(f"Wczytywanie plików z katalogu: {self.data_path}")
            for filename in os.listdir(self.data_path):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(self.data_path, filename)
                    logger.info(f"Ładowanie: {filename}")
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
        else:
            logger.error(f"Ścieżka {self.data_path} jest nieprawidłowa lub nie zawiera plików PDF.")
            raise ValueError(f"Nieprawidłowa ścieżka do danych: {self.data_path}")

        logger.info(f"Pomyślnie załadowano {len(documents)} stron z dokumentów.")
        return documents