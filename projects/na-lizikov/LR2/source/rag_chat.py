import requests
import os
import sys

# Добавляем путь к scripts для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import hybrid_search

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"


def call_ollama(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Ошибка при обращении к Ollama: {e}"


def ask(question, top_k=5):
    """
    Формирует ответ на вопрос пользователя используя RAG-пайплайн
    """
    # 1. Поиск релевантных чанков
    hits = hybrid_search(question, top_k)

    if not hits:
        return "Извините, не удалось найти релевантную информацию в документации."

    # 2. Сборка контекста с метаданными
    contexts = []
    sources = []
    
    for i, h in enumerate(hits, 1):
        src = h["_source"]
        score = h.get("_score", 0)
        
        # Формируем сниппет (первые 500 символов)
        text = src.get("text", "")
        snippet = text[:500].replace("\n", " ").strip()
        if len(text) > 500:
            snippet += "..."
        
        # Извлекаем информацию об источнике
        source = src.get("source", "Неизвестный источник")
        relative_path = src.get("relative_path", "")
        
        contexts.append(
            f"[Контекст {i}] (релевантность: {score:.2f})\n"
            f"Источник: {source}\n"
            f"Путь: {relative_path}\n"
            f"Текст: {snippet}\n"
        )
        
        sources.append({
            "source": source,
            "path": relative_path,
            "score": score
        })

    # 3. Формирование промпта
    prompt = f"""Ты — ассистент, отвечающий строго по документации Python.
Используй ТОЛЬКО информацию из предоставленного контекста ниже.
Если в контексте нет ответа на вопрос, честно скажи об этом.

Контекст из документации:
{chr(10).join(contexts)}

Вопрос пользователя:
{question}

Инструкции:
- Ответь на вопрос, используя только информацию из контекста
- Если информация отсутствует в контексте, скажи об этом
- В конце ответа обязательно укажи источники (источник и путь)
- Будь точным и конкретным

Ответ:"""

    # 4. Вызов LLM
    answer = call_ollama(prompt)
    
    # 5. Формирование ответа с цитатами
    sources_text = "\n\nИсточники:\n"
    for src in sources:
        sources_text += f"- {src['source']} ({src['path']})\n"
    
    return answer + sources_text


if __name__ == "__main__":
    print("RAG-чат запущен. Введите 'exit' для выхода.\n")
    while True:
        try:
            q = input("Вопрос: ").strip()
            if q.lower() in ['exit', 'quit', 'выход']:
                print("До свидания!")
                break
            if not q:
                continue
            print("\n" + "="*50)
            print(ask(q))
            print("="*50 + "\n")
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}\n")
