import os
import requests
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Подавление предупреждений от внутренних библиотек (для чистоты консоли)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Конфигурация ---
INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
KOBOLD_API_URL = "http://localhost:5001/api/v1/generate" # тут я свое ставлю, реально

def load_vector_db():
    """
    Загружает ранее созданный векторный индекс FAISS.
    Использует ту же модель эмбеддингов, что и при создании базы.
    """
    print("[*] Загрузка модели эмбеддингов и индекса FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Разрешаем опасную десериализацию, так как индекс создан нами локально
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("[+] Индекс успешно загружен!")
    return vectorstore

def create_prompt(query, chunks):
    """
    Формирует итоговый промпт для LLM, объединяя:
    1. Системную инструкцию
    2. Few-shot примеры (из нашей синтетической базы)
    3. Chain-of-Thought инструкцию (CoT)
    4. Найденный контекст
    5. Запрос пользователя
    """
    # Собираем текст из найденных чанков
    context = "\n\n---\n\n".join([c.page_content for c in chunks])
    
    prompt = f"""You are a precise analytical AI assistant. Answer the User's question based strictly on the Context below. 
If the answer is not in the Context, reply EXACTLY with: "Я не знаю." (I don't know). Do not invent facts.

Use Chain-of-Thought (CoT) reasoning. First, explain your logic step-by-step starting with "Thought:". Then, provide your final answer starting with "Answer:".

=== FEW-SHOT EXAMPLE ===
Context: 
The Star Strider is a highly modified light freighter flown by Jax Rigger and his Ursine-Humanoid first mate, Krull the Tall. It played a key role in the Fringe Resistance.

User: Who is the first mate on the Star Strider?
Thought:
1. The user asks about the first mate of the Star Strider.
2. I scan the context for "Star Strider" and "first mate".
3. The context says it is flown by Jax Rigger and his Ursine-Humanoid first mate, Krull the Tall.
4. Therefore, the first mate is Krull the Tall.
Answer: The first mate on the Star Strider is Krull the Tall.
=== END OF EXAMPLE ===

=== REAL QUERY ===
Context:
{context}

User: {query}
Thought:"""
    return prompt

def ask_llm_kobold(prompt):
    """
    Отправляет сформированный промпт в локальную LLM через KoboldCPP API.
    Использует низкую температуру для минимизации галлюцинаций (0.1).
    """
    payload = {
        "prompt": prompt,
        "max_context_length": 4096, # Вот туточки можно потом и добавить
        "max_length": 512,  # Ограничение на длину ответа
        "temperature": 0.1, # Низкая температура для строгого следования фактам
        "top_p": 0.9,
        "rep_pen": 1.1,
        "stop_sequence": ["User:", "\n\n\n", "==="] # Триггеры для остановки генерации
    }
    
    try:
        response = requests.post(KOBOLD_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        # Извлекаем сгенерированный текст из JSON ответа KoboldCPP
        result_text = response.json()["results"][0]["text"].strip()
        return result_text
    except requests.exceptions.ConnectionError:
        return "Ошибка: Не удалось подключиться к KoboldCPP. Проверьте, запущен ли сервер на порту 5001."
    except Exception as e:
        return f"Ошибка API: {e}"

def main():
    """
    Основной цикл консольного приложения (REPL интерфейс).
    """
    print("="*50)
    print("Инициализация RAG-бота (QuantumForge Software)")
    print("="*50)
    
    # 1. Загрузка базы
    try:
        vectorstore = load_vector_db()
    except Exception as e:
        print(f"[-] Ошибка загрузки индекса: {e}")
        print("Убедитесь, что папка faiss_index существует и скрипт build_index.py был выполнен.")
        return

    print("\nБот готов! Введите ваш вопрос (или 'exit' для выхода).")
    
    # 2. Интерактивный цикл (REPL)
    while True:
        try:
            query = input("\n[Вы]: ")
            if query.lower() in ['exit', 'quit', 'выход']:
                break
            if not query.strip():
                continue
                
            # Шаг A: Иск ближайших документов (k=3 для достаточного контекста)
            docs = vectorstore.similarity_search(query, k=3)
            
            # Шаг B: Формирование промпта (Few-shot + CoT)
            prompt = create_prompt(query, docs)
            
            # Шаг C: Отправка в LLM
            print("[Бот думает...]")
            answer = ask_llm_kobold(prompt)
            
            # Шаг D: Вывод результата
            print(f"\n[Бот]:\nThought: {answer}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[!] Внутренняя ошибка: {e}")

if __name__ == "__main__":
    import sys
    # Исправление кодировки консоли для Windows (тире и спецсимволы)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    main()