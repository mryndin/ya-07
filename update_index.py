import os
import time
import sys
import json
import argparse
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Конфигурация ---
KB_DIR = "knowledge_base"
INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REGISTRY_FILE = "kb_registry.json"
LOG_FILE = "update.log"

def log(message):
    """Логирование в файл и дублирование в консоль."""
    # Исправление ошибки кодировки в консоли Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def get_current_state():
    """Сканирует папку и возвращает словарь {имя_файла: mtime}."""
    state = {}
    if not os.path.exists(KB_DIR):
        return state
    for filename in os.listdir(KB_DIR):
        if filename.endswith(".md"):
            path = os.path.join(KB_DIR, filename)
            state[filename] = os.path.getmtime(path)
    return state

def build_and_save_index(run_tests=False):
    """Основная логика сборки индекса, перенесенная из оригинального build_index.py."""
    log("[*] Инициализация процесса создания векторного индекса...")
    
    # 1. Загрузка документов из папки
    # Используем TextLoader для чтения .md файлов, кодировка UTF-8
    loader = DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()
    log(f"[+] Загружено документов: {len(docs)}")

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
    log(f"[+] Документы разбиты на {len(chunks)} чанков.")

    # 3. Инициализация модели эмбеддингов
    # Загружается локально (первый запуск скачает веса ~90MB в кэш HuggingFace)
    log(f"[*] Загрузка модели эмбеддингов: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # 4. Создание векторного индекса FAISS
    log("[*] Генерация эмбеддингов и создание индекса FAISS (это может занять немного времени)...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    end_time = time.time()
    
    generation_time = end_time - start_time
    log(f"[+] Индекс создан за {generation_time:.2f} секунд.")

    # 5. Сохранение индекса на диск
    vectorstore.save_local(INDEX_PATH)
    log(f"[+] Индекс сохранен в директорию: {INDEX_PATH}")

    # --- ТЕСТИРОВАНИЕ ИНДЕКСА ---
    if run_tests:
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

def main():
    parser = argparse.ArgumentParser(description="Автоматическое обновление базы знаний RAG")
    parser.add_argument("--force", action="store_true", help="Принудительно пересобрать индекс, игнорируя реестр")
    parser.add_argument("--test", action="store_true", help="Запустить встроенные тесты после сборки")
    args = parser.parse_args()

    if not os.path.exists(KB_DIR):
        log(f"[-] Ошибка: Директория базы знаний '{KB_DIR}' не найдена. Создайте её и добавьте .md файлы.")
        return

    # 1. Читаем старое состояние (реестр)
    registry = {}
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            registry = json.load(f)

    # 2. Получаем текущее состояние из папки
    current_state = get_current_state()

    # 3. Ищем разницу (mtime или состав файлов)
    added = [f for f in current_state if f not in registry]
    deleted = [f for f in registry if f not in current_state]
    modified = [f for f in current_state if f in registry and current_state[f] > registry[f]]

    # 4. Принимаем решение о пересборке
    if args.force or added or deleted or modified:
        if args.force:
            log("[*] Триггер: ЗАПУСК ПРИНУДИТЕЛЬНОЙ ПЕРЕСБОРКИ (--force)")
        else:
            log(f"[*] Триггер: Обнаружены изменения (+{len(added)} нов, -{len(deleted)} уд, {len(modified)} изм).")
        
        # Пересобираем индекс
        build_and_save_index(run_tests=args.test)
        
        # 5. Сохраняем обновленный реестр только после успешной сборки
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(current_state, f, indent=4, ensure_ascii=False)
        
        log("[+] Реестр состояний успешно обновлен.")
    else:
        log("[+] Изменений не обнаружено. Индекс актуален. Пропуск сборки.")

if __name__ == "__main__":
    main()