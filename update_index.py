import os
import time
import sys
import json
import argparse
import threading
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Модули для блокировки файлов в зависимости от ОС
if sys.platform == "win32":
    import msvcrt
else:
    try:
        import fcntl
    except ImportError:
        fcntl = None

# --- КОНФИГУРАЦИЯ С АБСОЛЮТНЫМИ ПУТЯМИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REGISTRY_FILE = os.path.join(BASE_DIR, "kb_registry.json")
LOG_FILE = os.path.join(BASE_DIR, "update.log")
LOCK_FILE = os.path.join(BASE_DIR, "index.lock")

def log(message):
    """Логирование в файл и дублирование в консоль."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except: pass
        
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
    """Основная логика сборки индекса."""
    try:
        log("[*] Инициализация процесса создания векторного индекса...")
        
        loader = DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        docs = loader.load()
        log(f"[+] Загружено документов: {len(docs)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        log(f"[+] Документы разбиты на {len(chunks)} чанков.")

        log(f"[*] Загрузка модели эмбеддингов: {MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

        log("[*] Генерация эмбеддингов и создание индекса FAISS...")
        start_time = time.time()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        log(f"[+] Индекс создан за {time.time() - start_time:.2f} секунд.")

        vectorstore.save_local(INDEX_PATH)
        log(f"[+] Индекс сохранен в директорию: {INDEX_PATH}")

        if run_tests:
            test_queries = ["Who is Xarn Velgor?", "What is the power of Synth Flux?"]
            for query in test_queries:
                results = vectorstore.similarity_search(query, k=2)
                log(f"[Тест] Запрос: '{query}'. Найдено результатов: {len(results)}")
        return True
    except Exception as e:
        log(f"[!!!] ОШИБКА ВНУТРИ ПРОЦЕССА: {e}")
        return False

def main():
    print(f"[+] Скрипт начал выполнение.")
    lock_file_handle = open(LOCK_FILE, "w")
    
    try:
        # 1. Захват монопольного доступа
        if sys.platform == "win32":
            try:
                msvcrt.locking(lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
            except IOError:
                print(f"[!] Скрипт уже выполняется другим процессом. Выход.")
                sys.exit(0)
        else:
            if fcntl:
                try:
                    fcntl.flock(lock_file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (IOError, BlockingIOError):
                    print(f"[!] Скрипт уже выполняется. Выход.")
                    sys.exit(0)

        lock_file_handle.write(str(os.getpid()))
        lock_file_handle.flush()

        # 2. Парсинг аргументов
        parser = argparse.ArgumentParser(description="Автоматическое обновление базы знаний RAG")
        parser.add_argument("--force", action="store_true", help="Принудительно пересобрать индекс")
        parser.add_argument("--test", action="store_true", help="Запустить тесты")
        parser.add_argument("--retries", type=int, default=3, help="Количество повторов при ошибке")
        parser.add_argument("--timeout", type=int, default=3600, help="Таймаут выполнения одной попытки (сек)")
        args = parser.parse_args()

        if not os.path.exists(KB_DIR):
            log(f"[-] Ошибка: Директория '{KB_DIR}' не найдена.")
            return

        # 3. Проверка изменений
        registry = {}
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                registry = json.load(f)

        current_state = get_current_state()
        added = [f for f in current_state if f not in registry]
        deleted = [f for f in registry if f not in current_state]
        modified = [f for f in current_state if f in registry and current_state[f] > registry[f]]

        # 4. Выполнение обновления с Retries и Timeout
        if args.force or added or deleted or modified:
            log(f"[*] Триггер: Обнаружены изменения (+{len(added)}, -{len(deleted)}, mod:{len(modified)}).")
            
            success = False
            for i in range(1, args.retries + 1):
                log(f"[*] Попытка {i} из {args.retries} (Таймаут: {args.timeout}с)...")
                
                # Запуск в отдельном потоке для контроля времени
                result_container = {"status": False}
                
                def worker():
                    result_container["status"] = build_and_save_index(run_tests=args.test)

                thread = threading.Thread(target=worker)
                thread.start()
                thread.join(timeout=args.timeout)

                if thread.is_alive():
                    log(f"[!] Попытка {i} прервана по таймауту ({args.timeout}с).")
                    # Поток останется в фоне до завершения, но мы идем дальше
                elif result_container["status"]:
                    log(f"[+] Попытка {i} завершена успешно.")
                    success = True
                    break
                else:
                    log(f"[-] Попытка {i} завершилась с ошибкой.")

            if success:
                with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                    json.dump(current_state, f, indent=4, ensure_ascii=False)
                log("[+] Индекс и реестр успешно обновлены.")
            else:
                log("[!!!] Не удалось обновить индекс после всех попыток.")
        else:
            print("[+] Изменений не обнаружено. Пропуск.")

    except Exception as e:
        log(f"[!!!] КРИТИЧЕСКАЯ ОШИБКА: {e}")
    finally:
        if sys.platform == "win32":
            try:
                lock_file_handle.seek(0)
                msvcrt.locking(lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1)
            except: pass
        lock_file_handle.close()
        print(f"[+] Скрипт окончил выполнение.")

if __name__ == "__main__":
    main()