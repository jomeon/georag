# System RAG dla Domeny Geoinformatycznej (GeoRAG)

## 📌 Cel Projektu
Celem projektu jest budowa zaawansowanego systemu Retrieval-Augmented Generation (RAG) wspomagającego analizę i ekstrakcję wiedzy z literatury naukowej oraz dokumentacji technicznej z zakresu geoinformatyki (GIS, teledetekcja, fotogrametria). System odpowiada na zapytania użytkownika w języku naturalnym, opierając się wyłącznie na dostarczonym korpusie dokumentów (plikach PDF), minimalizując ryzyko halucynacji modelu językowego.

## 🏗️ Architektura Systemu i Uzasadnienie Decyzji (Sprawozdanie)

Projekt został zrealizowany w paradygmacie obiektowym (OOP) przy użyciu języka Python i frameworka LangChain. Architektura dzieli się na pięć niezależnych modułów:

### 1. Przetwarzanie i Fragmentacja (Chunking)
*   **Moduł:** `src/document_loader.py` oraz `src/chunker.py`
*   **Technologia:** `PyPDFLoader`, `RecursiveCharacterTextSplitter`.
*   **Uzasadnienie:** Literatura geoinformatyczna zawiera złożone definicje i opisy układów odniesienia. Zastosowano `RecursiveCharacterTextSplitter`, aby zapobiec przecinaniu zdań i akapitów w połowie. Wybrano rozmiar fragmentu (**chunk_size**) na 1000 znaków, co pozwala zmieścić średniej długości akapit naukowy. Zakładka (**chunk_overlap**) wynosi 200 znaków – stanowi ona bufor bezpieczeństwa, dzięki któremu kontekst nie zostaje zerwany na granicy dwóch fragmentów tekstu.

### 2. Embeddingi i Baza Wektorowa (Retriever)
*   **Moduł:** `src/retriever.py`
*   **Technologia:** Baza `ChromaDB`, model embeddingowy `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
*   **Uzasadnienie:** Wybrano wielojęzyczny model z HuggingFace, ponieważ polskie teksty o GIS często zawierają angielską terminologię (np. *Remote Sensing*, *Point Cloud*). Model ten jest wydajny i działa płynnie lokalnie na CPU. Do przechowywania wektorów wybrano bazę `ChromaDB`, która (w przeciwieństwie np. do czystego indeksu FAISS) natywnie zarządza metadanymi. W systemach dziedzinowych krytyczne jest wskazanie źródła informacji – Chroma automatycznie przechowuje nazwę pliku PDF oraz numer strony dla każdego chunk'a.

### 3. Generator Odpowiedzi (LLM)
*   **Moduł:** `src/generator.py`
*   **Technologia:** Google Gemini .5 Flash (`langchain-google-genai`).
*   **Uzasadnienie:** Zastosowano model Gemini ze względu na jego dużą wydajność i ogromne okno kontekstowe. Temperaturę modelu ustawiono na `0.0` (pełen determinizm), co jest kluczowe w systemach eksperckich. Zastosowano rygorystyczny *System Prompt* zmuszający model do odpowiadania wyłącznie na podstawie dostarczonych dokumentów. Wyszukiwanie wektorowe opiera się na 4 najbardziej zbliżonych fragmentach (`k=4`), aby nie rozpraszać modelu nadmiarem szumu informacyjnego.

### 4. Testy i Ewaluacja (LLM-as-a-Judge)
*   **Moduł:** `tests/` oraz `src/evaluator.py`
*   **Uzasadnienie:** Kod produkcyjny jest pokryty testami jednostkowymi (`pytest`). Ze względu na niedeterministyczną naturę odpowiedzi generowanych przez LLM, wdrożono system ewaluacji oparty na koncepcji "LLM jako sędzia" inspirowany frameworkiem RAGAS. System ocenia dwie krytyczne metryki:
    1.  **Faithfulness (Wierność):** Weryfikuje, czy wygenerowana odpowiedź posiada pokrycie w faktach dostarczonych w dokumentach źródłowych (detekcja halucynacji).
    2.  **Answer Relevance (Trafność):** Mierzy, w jakim stopniu odpowiedź odpowiada na faktyczną intencję zapytania użytkownika.

## 🚀 Instrukcja Uruchomienia

1.  **Sklonowanie i środowisko:**
    ```bash
    conda create -n georag python=3.10 -y
    conda activate georag
    pip install -r requirements.txt
    ```

2.  **Konfiguracja kluczy:**
    *   Utwórz plik `.env` w głównym katalogu projektu.
    *   Wpisz swój klucz API: `GOOGLE_API_KEY=twój_klucz_tutaj`

3.  **Dane:**
    *   Umieść pliki PDF z zakresu geoinformatyki w folderze `data/raw/`.

4.  **Uruchomienie:**
    *   Otwórz plik `notebooks/demo_rag.ipynb` w Jupyter Notebook lub VS Code.
    *   Uruchom komórki po kolei, aby zbudować bazę i przetestować zapytania.

5.  **Testy:**
    ```bash
    pytest tests/
    ```
