import os
import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Конфигурация ---
KB_DIR = "knowledge_base"
INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("[*] Инициализация процесса создания векторного индекса...")
    
    # 1. Загрузка документов из папки
    # Используем TextLoader для чтения .md файлов, кодировка UTF-8
    loader = DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()
    print(f"[+] Загружено документов: {len(docs)}")

    # 2. Разбиение на чанки (Chunking)
    # chunk_size=1000 символов (~200-250 токенов/слов). 
    # chunk_overlap=200 символов для сохранения контекста на разрывах.
    # add_start_index=True сохранит метаданные о позиции чанка в исходном файле.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[+] Документы разбиты на {len(chunks)} чанков.")

    # 3. Инициализация модели эмбеддингов
    # Загружается локально (первый запуск скачает веса ~90MB в кэш HuggingFace)
    print(f"[*] Загрузка модели эмбеддингов: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # 4. Создание векторного индекса FAISS
    print("[*] Генерация эмбеддингов и создание индекса FAISS (это может занять немного времени)...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    end_time = time.time()
    
    generation_time = end_time - start_time
    print(f"[+] Индекс создан за {generation_time:.2f} секунд.")

    # 5. Сохранение индекса на диск
    vectorstore.save_local(INDEX_PATH)
    print(f"[+] Индекс сохранен в директорию: {INDEX_PATH}")

    # --- ТЕСТИРОВАНИЕ ИНДЕКСА ---
    print("\n" + "="*40)
    print("ТЕСТИРОВАНИЕ ПОИСКА ПО ИНДЕКСУ")
    print("="*40)
    
    test_queries = [
        "Who is Xarn Velgor?",                  # Проверка подмены Дарта Вейдера
        "What is the power of Synth Flux?"      # Проверка подмены Силы
    ]

    for query in test_queries:
        print(f"\nЗапрос: '{query}'")
        # Ищем 2 самых релевантных чанка
        results = vectorstore.similarity_search(query, k=2)
        
        for i, res in enumerate(results):
            source = res.metadata.get('source', 'Unknown')
            start_idx = res.metadata.get('start_index', 'Unknown')
            print(f"\n--- Чанк {i+1} (Источник: {source}, Позиция: {start_idx}) ---")
            print(res.page_content.strip())
            print("-" * 50)

if __name__ == "__main__":
    main()