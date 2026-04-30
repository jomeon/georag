import os
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

logger = logging.getLogger(__name__)

class RAGGenerator:
    """
    Klasa odpowiedzialna za łączenie bazy wektorowej (Retriever) 
    z modelem językowym (LLM) w celu generowania odpowiedzi opartych na kontekście.
    """

    def __init__(self, vector_store, model_name: str = "gemini-flash", temperature: float = 0.0):
        """
        Inicjalizuje generator RAG.
        
        Args:
            vector_store: Instancja bazy Chroma z załadowanymi dokumentami.
            model_name (str): Nazwa modelu. Używamy wydajnego 'gemini-1.5-flash'.
            temperature (float): Poziom kreatywności modelu. 0.0 oznacza maksymalną deterministyczność 
                                 (idealne dla systemów naukowych - zapobiega halucynacjom).
        """
        self.vector_store = vector_store
        
        logger.info(f"Inicjalizacja modelu LLM: {model_name} (temperature={temperature})")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        self.rag_chain = self._build_chain()

    def _build_chain(self):
        """
        Prywatna metoda budująca łańcuch LangChain z rygorystycznym promptem.
        """
        system_prompt = (
            "Jesteś eksperckim asystentem AI z dziedziny geoinformatyki. "
            "Twoim zadaniem jest odpowiadanie na pytania użytkownika WYŁĄCZNIE na podstawie "
            "dostarczonego poniżej kontekstu. \n\n"
            "ZASADY:\n"
            "1. Jeśli odpowiedź nie znajduje się w kontekście, powiedz wprost: 'Niestety, "
            "nie znalazłem informacji na ten temat w dostarczonych dokumentach.' Nie wymyślaj odpowiedzi.\n"
            "2. Odpowiadaj zwięźle, w języku polskim, zachowując naukowy i profesjonalny ton.\n\n"
            "KONTEKST:\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        
        return rag_chain

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Zadaje pytanie do systemu RAG i zwraca odpowiedź wraz ze źródłami.
        """
        logger.info(f"Zadaję pytanie: '{question}'")
        response = self.rag_chain.invoke({"input": question})
        
        logger.info("Odpowiedź wygenerowana pomyślnie. Znalezione źródła:")
        for i, doc in enumerate(response.get("context", [])):
            source = doc.metadata.get("source", "Nieznane")
            page = doc.metadata.get("page", "Nieznana")
            logger.info(f"  [{i+1}] Plik: {source}, Strona: {page}")
            
        return response