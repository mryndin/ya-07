import sys
import requests
import argparse
import datetime
import re
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Подавление предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)

# --- КОНФИГУРАЦИЯ (Экспортируется для evaluate.py) ---
INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = "http://qwen.rs-soft.site/v1/chat/completions"
NO_ANSWER = "Я не знаю."

# --- СЛОИ ЗАЩИТЫ ---

def sanitize_content(text):
    patterns = [r"(?i)ignore\s+all\s+instructions", r"(?i)system\s+override"]
    for p in patterns:
        text = re.sub(p, "[REDACTED]", text)
    return text

def is_unsafe(text):
    forbidden = ["swordfish", "root:", "superpassword"]
    return any(word in text.lower() for word in forbidden)

def create_prompt(query, chunks, security_enabled):
    """Гибридный промпт без лишних рассуждений (CoT)."""
    if not chunks:
        context_text = "ИНФОРМАЦИЯ В БАЗЕ ЗНАНИЙ ОТСУТСТВУЕТ."
    else:
        context_list = [sanitize_content(c.page_content) if security_enabled else c.page_content for c in chunks]
        context_text = "\n\n---\n\n".join(context_list)

    return f"""You are a precise analytical assistant.
1. Use the Context to answer.
2. If Context is missing or insufficient, use your OWN internal knowledge.
3. If you truly don't know, reply EXACTLY: "{NO_ANSWER}"

STRICT: No reasoning. No "Based on...". Direct answer only.

=== CONTEXT ===
{context_text}

=== USER QUERY ===
{query}

=== ANSWER ===
"""

# --- ОСНОВНЫЕ ФУНКЦИИ ---

def ask_llm(prompt):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))

    payload = {
        "model": "Dolphin-Mistral",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 350,
        "stop": ["User:", "==="]
    }
    
    try:
        response = session.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        # Очистка от остаточного "мусора"
        if NO_ANSWER.lower() in answer.lower() and len(answer) < 30:
            return NO_ANSWER
        return answer
    except Exception as e:
        return f"Ошибка API: {e}"

def process_query(query, vectorstore, security_enabled, log_file):
    """Основная функция для бота и evaluate.py."""
    # Поиск (используем similarity_search_with_score для фильтрации шума)
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)
    
    # Считаем релевантными только те, где score < 1.0
    relevant_docs = [doc for doc, score in docs_and_scores if score < 1.0]
    
    status = "HYBRID_SUCCESS"
    if security_enabled and relevant_docs:
        relevant_docs = [d for d in relevant_docs if not is_unsafe(d.page_content)]
        if not relevant_docs: status = "SECURITY_FILTERED"

    prompt = create_prompt(query, relevant_docs, security_enabled)
    answer = ask_llm(prompt)

    # Логирование
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] SEC: {security_enabled} | Q: {query} | A: {answer}\n")
        
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str)
    parser.add_argument("--no-security", action="store_true")
    parser.add_argument("--log", type=str, default="bot_history.log")
    args = parser.parse_args()

    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[-] Ошибка: {e}")
        return

    sec_on = not args.no_security
    if args.query:
        print(f"\n[Бот]: {process_query(args.query, vectorstore, sec_on, args.log)}")
    else:
        print(f"RAG-бот готов (Защита: {'ВКЛ' if sec_on else 'ВЫКЛ'})")
        while True:
            q = input("\n[Вы]: ")
            if q.lower() in ['exit', 'quit']: break
            print(f"\n[Бот]: {process_query(q, vectorstore, sec_on, args.log)}")

if __name__ == "__main__":
    main()