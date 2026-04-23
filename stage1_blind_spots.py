import os
import shutil
import sys
import subprocess
from datetime import datetime

# --- КОНФИГУРАЦИЯ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
SHADOW_DIR = os.path.join(BASE_DIR, "knowledge_base_shadow")
UPDATE_SCRIPT = os.path.join(BASE_DIR, "update_index.py")

# Сущности для удаления (эмуляция "слепых зон")
ENTITIES_TO_REMOVE = ["Xarn Velgor", "VoidCore", "Synth Flux"]

def log(message):
    """Консольное логирование с принудительным UTF-8."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except: pass
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def create_shadow_copy():
    """Создает бэкап базы знаний, если он еще не создан."""
    if os.path.exists(SHADOW_DIR):
        log(f"[*] Shadow-копия уже существует по пути: {SHADOW_DIR}. Пропуск.")
        return True
    try:
        shutil.copytree(KB_DIR, SHADOW_DIR)
        log("[+] Резервная копия создана.")
        return True
    except Exception as e:
        log(f"[-] Ошибка при создании резервной копии: {e}")
        return False

def inject_blind_spots():
    """Удаляет строки с сущностями из файлов .md, используя UTF-8."""
    log(f"[*] Инициализация удаления сущностей: {', '.join(ENTITIES_TO_REMOVE)}")
    
    modified_files_count = 0
    removed_lines_total = 0

    for root, _, files in os.walk(KB_DIR):
        for file in files:
            if not file.endswith(".md"):
                continue
            
            filepath = os.path.join(root, file)
            
            # ЧТЕНИЕ: Явно указываем кодировку utf-8
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                log(f"[-] Ошибка кодировки в файле {file}, пропуск.")
                continue

            new_lines = []
            lines_removed_in_file = 0
            
            for line in lines:
                if any(entity.lower() in line.lower() for entity in ENTITIES_TO_REMOVE):
                    lines_removed_in_file += 1
                    removed_lines_total += 1
                else:
                    new_lines.append(line)

            # ЗАПИСЬ: Сохраняем результат
            if lines_removed_in_file > 0:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                log(f"[~] Обновлен файл: {file} (-{lines_removed_in_file} строк)")
                modified_files_count += 1

    log(f"[+] Изменено файлов: {modified_files_count}, удалено упоминаний: {removed_lines_total}.")

def trigger_index_rebuild():
    """Вызывает update_index.py с безопасной обработкой потока вывода."""
    log("[*] Запуск пересборки (update_index.py --force)...")
    try:
        # ЗАПУСК: text=False (читаем байты) для предотвращения UnicodeDecodeError
        result = subprocess.run(
            [sys.executable, UPDATE_SCRIPT, "--force"],
            capture_output=True,
            text=False 
        )
        
        # ДЕКОДИРОВАНИЕ: Ручное с обработкой ошибок 'replace'
        # Битые символы заменятся на '?', скрипт не упадет
        stdout_text = result.stdout.decode('utf-8', errors='replace')
        stderr_text = result.stderr.decode('utf-8', errors='replace')
        
        # Вывод результатов
        if stdout_text:
            for line in stdout_text.splitlines():
                print(f"    | {line}")
        
        if result.returncode == 0:
            log("[+] Индекс успешно пересобран.")
        else:
            log(f"[-] Ошибка пересборки (код {result.returncode})")
            if stderr_text:
                print(f"    ! {stderr_text}")
            
    except Exception as e:
        log(f"[-] Критическая ошибка при вызове скрипта: {e}")

def main():
    if not os.path.exists(KB_DIR):
        log("[-] Папка knowledge_base не найдена.")
        sys.exit(1)

    if create_shadow_copy():
        inject_blind_spots()
        trigger_index_rebuild()
    
    log("[+] Этап 1 завершен.")

if __name__ == "__main__":
    main()