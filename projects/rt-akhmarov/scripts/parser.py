import requests
import os
import re

BASE_URL = "https://verl.readthedocs.io/en/latest/"
SOURCES_URL = BASE_URL + "_sources/"
OUTPUT_DIR = "verl_sources"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_page_list():
    response = requests.get(BASE_URL + "searchindex.js")
    if response.status_code != 200:
        print("Не удалось получить индекс страниц.")
        return []
    
    match = re.search(r'"filenames":\[(.*?)\]', response.text)
    if match:
        filenames = match.group(1).replace('"', '').split(',')
        return filenames
    return []

def download_sources():
    pages = get_page_list()
    if not pages:
        print("Список страниц пуст.")
        return

    print(f"Найдено страниц: {len(pages)}")

    for page in pages:
        source_url = f"{SOURCES_URL}{page}.txt"
        try:
            res = requests.get(source_url)
            if res.status_code == 200:
                file_path = os.path.join(OUTPUT_DIR, f"{page.replace('/', '_')}")
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(res.text)
                print(f"Сохранено: {page}")
            else:
                print(f"Пропущено (404): {source_url}")
        except Exception as e:
            print(f"Ошибка при загрузке {page}: {e}")

if __name__ == "__main__":
    download_sources()