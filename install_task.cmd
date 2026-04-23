@echo off
setlocal enabledelayedexpansion

:: Определяем путь к текущей папке и интерпретатору Python
set "SCRIPT_PATH=%~dp0update_index.py"
set "PYTHON_EXE=python.exe"

:: Название задачи в планировщике
set "TASK_NAME=RAG_KnowledgeBase_Update"

echo [*] Регистрация задачи в Планировщике Windows...
echo [!] Скрипт: %SCRIPT_PATH%

:: Создаем задачу (ежедневно в 01:00)
:: /sc daily - ежедневно
:: /st 01:00 - время запуска
:: /tr - команда (запускаем python со скриптом)
:: /f - принудительная перезапись, если задача уже есть
:: /du 02:00 - ограничение длительности (2 часа)
:: /ri 120 - принудительное завершение задачи, если она превысила лимит
schtasks /create /tn "%TASK_NAME%" /tr "%PYTHON_EXE% \"%SCRIPT_PATH%\"" /sc daily /st 01:00 /du 02:00 /ri 90 /f

if %errorlevel% equ 0 (
    echo [+] Задача успешно создана!
    echo [*] Теперь скрипт будет проверять обновления KB каждую ночь.
) else (
    echo [-] Ошибка при создании задачи. Попробуйте запустить CMD от имени Администратора.
)

pause