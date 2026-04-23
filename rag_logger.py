import os
import json
from datetime import datetime

class QueryLogger:
    """
    Модуль для логирования запросов к RAG-системе.
    Сохраняет данные в формате JSONL (одна запись - одна строка валидного JSON).
    """
    def __init__(self, log_filename="logs.jsonl"):
        # --- КОНФИГУРАЦИЯ С АБСОЛЮТНЫМИ ПУТЯМИ ---
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = os.path.join(self.base_dir, "logs")
        
        # Создаем папку логов, если ее нет
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.logs_dir, log_filename)

    def _evaluate_status(self, has_chunks: bool, response: str) -> str:
        """
        Базовая эвристика для оценки успешности ответа.
        """
        response_lower = response.lower()
        # Типичные фразы LLM, когда она не знает ответа
        negative_keywords = ["не знаю", "нет информации", "не найдено", "не упоминается", "i don't know"]
        is_negative_response = any(kw in response_lower for kw in negative_keywords)

        if has_chunks:
            # Контекст найден. Если ответ не содержит явных отказов, считаем успешным.
            return "Success" if not is_negative_response else "Failed_To_Extract"
        else:
            # Контекст НЕ найден.
            # Если бот честно признался, что не знает (или ответ очень короткий) — это правильное поведение (True Negative).
            # Если бот выдал длинный текст без контекста — это высокий риск галлюцинации.
            if is_negative_response or len(response) < 50:
                return "Correctly_Rejected"
            else:
                return "Hallucination_Risk"

    def log_interaction(self, query: str, response: str, sources: list, latency: float = 0.0) -> dict:
        """
        Формирует запись о запросе и сохраняет ее в лог-файл.
        
        :param query: Текст запроса пользователя.
        :param response: Ответ от LLM.
        :param sources: Список имен файлов/источников, переданных в контекст.
        :param latency: Время генерации ответа (секунды).
        """
        has_chunks = len(sources) > 0
        response_len = len(response)
        status_flag = self._evaluate_status(has_chunks, response)

        log_entry = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "query": query,
            "has_chunks": has_chunks,
            "sources": sources,
            "response_len": response_len,
            "latency": round(latency, 3),
            "status_flag": status_flag,
            "response": response
        }

        # Дописываем строку в файл с принудительным UTF-8
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return log_entry

# Пример использования (можно удалить или закомментировать при импорте):
if __name__ == "__main__":
    logger = QueryLogger()
    
    # Имитация успешного запроса
    logger.log_interaction(
        query="Где находится база повстанцев?",
        response="База повстанцев находится на ледяной планете Хот.",
        sources=["Hoth.md", "Alliance.md"],
        latency=2.15
    )
    
    # Имитация запроса в "слепую зону" (бот правильно ответил, что не знает)
    logger.log_interaction(
        query="Кто такой Xarn Velgor?",
        response="К сожалению, в базе данных нет информации о персонаже Xarn Velgor.",
        sources=[],
        latency=0.85
    )
    
    # Имитация галлюцинации (контекста нет, но бот придумал ответ)
    logger.log_interaction(
        query="Что такое Synth Flux?",
        response="Synth Flux — это древняя энергия ситхов, позволяющая управлять временем.",
        sources=[],
        latency=3.40
    )
    
    print(f"[*] Тестовые логи записаны в {logger.log_file}")