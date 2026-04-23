import os
import json
import re
import time
import cloudscraper
from bs4 import BeautifulSoup

# --- Конфигурация ---
KB_DIR = "knowledge_base"
TERMS_MAP_FILE = "terms_map.json"

# Список страниц для парсинга
URLS = [
    "https://starwars.fandom.com/wiki/Darth_Vader", "https://starwars.fandom.com/wiki/Luke_Skywalker",
    "https://starwars.fandom.com/wiki/The_Force", "https://starwars.fandom.com/wiki/Lightsaber",
    "https://starwars.fandom.com/wiki/Jedi", "https://starwars.fandom.com/wiki/Sith",
    "https://starwars.fandom.com/wiki/Death_Star", "https://starwars.fandom.com/wiki/Millennium_Falcon",
    "https://starwars.fandom.com/wiki/Tatooine", "https://starwars.fandom.com/wiki/R2-D2",
    "https://starwars.fandom.com/wiki/C-3PO", "https://starwars.fandom.com/wiki/Yoda",
    "https://starwars.fandom.com/wiki/Palpatine", "https://starwars.fandom.com/wiki/Obi-Wan_Kenobi",
    "https://starwars.fandom.com/wiki/Leia_Organa", "https://starwars.fandom.com/wiki/Han_Solo",
    "https://starwars.fandom.com/wiki/Chewbacca", "https://starwars.fandom.com/wiki/Galactic_Empire",
    "https://starwars.fandom.com/wiki/Alliance_to_Restore_the_Republic", "https://starwars.fandom.com/wiki/Stormtrooper",
    "https://starwars.fandom.com/wiki/X-wing_starfighter", "https://starwars.fandom.com/wiki/TIE/ln_space_superiority_starfighter",
    "https://starwars.fandom.com/wiki/Hoth", "https://starwars.fandom.com/wiki/Endor",
    "https://starwars.fandom.com/wiki/Coruscant", "https://starwars.fandom.com/wiki/Boba_Fett",
    "https://starwars.fandom.com/wiki/Kylo_Ren", "https://starwars.fandom.com/wiki/Rey",
    "https://starwars.fandom.com/wiki/Finn", "https://starwars.fandom.com/wiki/Snoke",
    "https://starwars.fandom.com/wiki/Clone_Wars"
]

def load_terms():
    with open(TERMS_MAP_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text_content(text):
    # Убираем пустые строки, лишние пробелы
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def replace_terms(text, terms_map):
    # Сортируем ключи по длине (сначала длинные), чтобы "Darth Vader" не превратилось в "Xarn Vader"
    sorted_terms = sorted(terms_map.keys(), key=len, reverse=True)
    
    for original in sorted_terms:
        replacement = terms_map[original]
        # Используем \b для границ слов, чтобы не ломать структуру
        pattern = re.compile(rf'\b{re.escape(original)}\b', re.IGNORECASE)
        text = pattern.sub(replacement, text)
    return text

def main():
    if not os.path.exists(KB_DIR):
        os.makedirs(KB_DIR)
        print(f"[*] Создана директория {KB_DIR}")
    
    terms_map = load_terms()
    scraper = cloudscraper.create_scraper()

    for url in URLS:
        print(f"[*] Загрузка: {url}")
        try:
            resp = scraper.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"[!] Ошибка {resp.status_code} на {url}")
                continue
                
            soup = BeautifulSoup(resp.content, 'html.parser')
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            
            if not content_div:
                print(f"[!] Не найден контент на странице {url}")
                continue
            
            # Удаляем мусор
            for junk in content_div(['table', 'aside', 'script', 'style', 'nav', 'sup']):
                junk.decompose()
            
            raw_text = content_div.get_text(separator='\n')
            
            # Обработка
            processed_text = replace_terms(raw_text, terms_map)
            final_text = clean_text_content(processed_text)
            
            # Сохранение
            filename = url.split('/')[-1] + ".md"
            with open(os.path.join(KB_DIR, filename), 'w', encoding='utf-8') as f:
                f.write(f"# Source: {url}\n\n{final_text}")
            
            print(f"[+] Сохранено: {filename}")
            time.sleep(2) # Пауза между запросами
            
        except Exception as e:
            print(f"[!] Критическая ошибка: {e}")

if __name__ == "__main__":
    main()