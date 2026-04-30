import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Klasa pełniąca rolę 'LLM-as-a-Judge'. Używa modelu językowego do
    oceniania jakości wygenerowanych odpowiedzi w systemie RAG.
    """

    def __init__(self, model_name: str = "gemini-flash"):
        """
        Inicjalizuje ewaluatora. Temperatura ustawiona na 0.0, ponieważ
        oczekujemy od 'sędziego' surowych, analitycznych i powtarzalnych ocen,
        a nie kreatywności.
        """
        logger.info("Inicjalizacja modułu Ewaluatora RAG...")
        self.judge_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

    def evaluate_faithfulness(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Ocenia, czy wygenerowana odpowiedź opiera się wyłącznie na dostarczonym kontekście.
        """
        eval_prompt = PromptTemplate(
            input_variables=["question", "context", "answer"],
            template=(
                "Jesteś bezstronnym sędzią oceniającym systemy AI.\n"
                "Twoim zadaniem jest ocena 'Wierności' (Faithfulness) odpowiedzi.\n\n"
                "Pytanie: {question}\n"
                "Kontekst źródłowy:\n{context}\n\n"
                "Odpowiedź do oceny: {answer}\n\n"
                "ZASADY OCENY:\n"
                "1. Sprawdź, czy każda informacja w 'Odpowiedzi do oceny' znajduje potwierdzenie w 'Kontekście źródłowym'.\n"
                "2. Jeśli odpowiedź zawiera informacje (halucynacje), których nie ma w kontekście, ocena to 0.\n"
                "3. Jeśli odpowiedź jest w pełni poparta kontekstem, ocena to 1.\n\n"
                "Zwróć wynik WYNIK: [0 lub 1] oraz UZASADNIENIE: [krótkie wyjaśnienie]."
            )
        )
        
        chain = eval_prompt | self.judge_llm
        logger.info("Ocenianie wierności (Faithfulness)...")
        result = chain.invoke({"question": question, "context": context, "answer": answer})
        return {"metric": "Faithfulness", "evaluation": result.content}

    def evaluate_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Ocenia, czy odpowiedź bezpośrednio i sensownie odpowiada na zadane pytanie.
        """
        eval_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template=(
                "Jesteś bezstronnym sędzią oceniającym systemy AI.\n"
                "Twoim zadaniem jest ocena 'Trafności' (Answer Relevance) odpowiedzi.\n\n"
                "Pytanie: {question}\n"
                "Odpowiedź do oceny: {answer}\n\n"
                "ZASADY OCENY:\n"
                "1. Oceń w skali od 0 do 1 (gdzie 0 to brak związku z pytaniem, a 1 to idealna odpowiedź na temat).\n"
                "2. Zignoruj fakt, czy odpowiedź jest prawdziwa. Oceniasz tylko to, czy adresuje intencję pytania.\n\n"
                "Zwróć wynik WYNIK: [wartość 0-1] oraz UZASADNIENIE: [krótkie wyjaśnienie]."
            )
        )
        
        chain = eval_prompt | self.judge_llm
        logger.info("Ocenianie trafności (Answer Relevance)...")
        result = chain.invoke({"question": question, "answer": answer})
        return {"metric": "Answer Relevance", "evaluation": result.content}