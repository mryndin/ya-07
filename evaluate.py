import json
import sys
import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Импортируем существующие функции из RAG_bot
from RAG_bot import process_query, MODEL_NAME, INDEX_PATH
from rag_logger import QueryLogger

def evaluate():
    # 1. Загрузка Golden Set
    try:
        with open("golden_set.json", "r", encoding="utf-8") as f:
            golden_set = json.load(f)
    except FileNotFoundError:
        print("[-] Ошибка: Файл golden_set.json не найден.")
        return

    # 2. Инициализация инструментов
    print("[*] Загрузка индекса...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[-] Ошибка загрузки базы: {e}")
        return

    logger = QueryLogger(log_filename="evaluation_results.jsonl")
    
    results = {"passed": 0, "failed": 0, "hallucinations": 0}
    
    print(f"[*] Начало тестирования ({len(golden_set)} вопросов)...")

    # 3. Цикл тестирования
    for item in golden_set:
        query = item['query']
        expected_type = item['type']
        
        print(f"\n[?] Query: {query}")
        
        # Получаем ответ через логику бота
        # Используем временный файл для логов, чтобы не смешивать с основными
        response = process_query(query, vectorstore, True, "eval_temp.log")
        
        # 4. Анализ результата
        # Для negative-вопросов бот должен ответить, что не знает
        is_negative_response = "я не знаю" in response.lower() or "i don't know" in response.lower() or "no information" in response.lower()
        
        is_success = False
        if expected_type == 'positive':
            if not is_negative_response:
                is_success = True
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        elif expected_type == 'negative':
            if is_negative_response:
                is_success = True
                results["passed"] += 1
            else:
                is_success = False
                results["hallucinations"] += 1 # Это галлюцинация, так как бот придумал ответ
        
        status = "PASS" if is_success else "FAIL"
        print(f"    Ответ: {response[:100]}...")
        print(f"    Статус: {status}")
        
        # Логируем результат
        logger.log_interaction(
            query=query,
            response=response,
            sources=[], # Для оценки можем не детализировать источники
            latency=0.0
        )

    # 5. Итоговый отчет
    print("\n" + "="*40)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print(f"Всего вопросов: {len(golden_set)}")
    print(f"Успешно (Pass): {results['passed']}")
    print(f"Ошибки поиска (Fail): {results['failed']}")
    print(f"Галлюцинации (Negative Response Failed): {results['hallucinations']}")
    print("="*40)

if __name__ == "__main__":
    evaluate()