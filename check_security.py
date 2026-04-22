import subprocess
import os

def run_batch(questions, log_file, no_security=False):
    print(f"[*] Запуск теста: {'БЕЗ ЗАЩИТЫ' if no_security else 'С ЗАЩИТОЙ'}")
    
    with open(log_file, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            cmd = ["python", "RAG_bot.py", "--query", q]
            if no_security:
                cmd.append("--no-security")
            
            # Запуск RAG_bot.py как внешнего процесса
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            f.write(f"Q: {q}\n")
            f.write(f"A: {result.stdout}\n")
            f.write("-" * 50 + "\n")
            print(f"[{i}/{len(questions)}] Вопрос обработан.")

def main():
    if not os.path.exists("security_questions.txt"):
        print("[-] Файл security_questions.txt не найден!")
        return

    with open("security_questions.txt", "r", encoding="utf-8") as f:
        questions = [q.strip() for q in f if q.strip()]

    # Запускаем оба прохода
    run_batch(questions, "security_vulnerable.log", no_security=True)
    run_batch(questions, "security_protected.log", no_security=False)
    
    print("\n[+] Тестирование завершено.")
    print("[+] Логи лежат в security_vulnerable.log и security_protected.log")

if __name__ == "__main__":
    main()