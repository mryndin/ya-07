import sys
import requests
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Подавление предупреждений для чистоты консоли
warnings.filterwarnings("ignore", category=FutureWarning)

# --- КОНФИГУРАЦИЯ ---
INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Адрес вашего сервера KoboldCPP
#API_URL = "http://192.168.0.100:5001/v1/chat/completions"
API_URL = "http://qwen.rs-soft.site/v1/chat/completions"


def load_vector_db():
    """Загрузка векторного индекса из локальной папки."""
    print("[*] Загрузка модели эмбеддингов и индекса FAISS...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # allow_dangerous_deserialization=True нужен, т.к. индекс создан локально
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("[+] Индекс успешно загружен!")
    return vectorstore


def create_prompt(query, chunks):
    """Формирование промпта с Few-shot и Chain-of-Thought."""
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


def ask_llm(prompt):
    """Отправка запроса в KoboldCPP через OpenAI-совместимый API."""
    payload = {
        "model": "Dolphin-Mistral",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "stop": ["User:", "\n\n\n", "==="]
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        # Стандартный путь для OpenAI API
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "Ошибка: Не удалось подключиться к KoboldCPP по адресу 192.168.0.100:5001."
    except Exception as e:
        return f"Ошибка API: {e}"


def main():
    """Главный цикл REPL."""
    # Исправление кодировки для Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 50)
    print("RAG-бот запущен (QuantumForge Software)")
    print("=" * 50)

    try:
        vectorstore = load_vector_db()
    except Exception as e:
        print(f"[-] Ошибка: {e}")
        return

    print("\nБот готов! Введите ваш вопрос (или 'exit' для выхода).")

    while True:
        try:
            query = input("\n[Вы]: ")
            if query.lower() in ['exit', 'quit', 'выход']:
                break
            if not query.strip():
                continue

            # Поиск
            docs = vectorstore.similarity_search(query, k=3)

            # Генерация
            prompt = create_prompt(query, docs)
            print("[Бот думает...]")
            answer = ask_llm(prompt)

            print(f"\n[Бот]:\nThought: {answer}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[!] Ошибка: {e}")


if __name__ == "__main__":
    main()