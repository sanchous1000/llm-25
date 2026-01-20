import requests
import time
import json
from typing import Dict, Any
from datetime import datetime

OLLAMA_API_URL = "http://127.0.0.1:11434/v1/chat/completions"

MODELS = {
    "qwen": "qwen2.5:7b",
    "llama": "llama3:8b",
    "mistral": "mistral:7b"
}

PROMPTS = {
    "P1_generation": {
        "description": "Генерация вежливого письма",
        "content": (
            "Напиши вежливое официальное письмо клиенту с извинениями за задержку доставки и обещанием компенсации. Язык письма: русский"
        )
    },
    "P2_classification": {
        "description": "Классификация обращений",
        "content": (
            "Определи категорию обращения клиента. Определенный список меток: "
            "[Billing, Tech support, Sales].\n\n"
            "Тексты для анализа: «У меня не проходит оплата по карте, деньги списались, но заказ не оформлен.»\n"
            "«У меня не запускается проект после последнего обновления вашей платформы. Выдает ошибку 404.»\n"
            "«Можете прислать актуальный прайс-лист на корпоративные тарифы?»\n"
            "«В моем счете за август есть строка за услугу 'Премиум-поддержка', которую я не подключал. Объясните, откуда она взялась.»\n"
            "Ответь в формате JSON-массива объектов, где каждый объект имеет ключи: `«id»` (порядковый номер, начиная с 1), `«text»` (исходный текст) и `«label»` (выбранная метка)."
        )
    },
    "P3_extraction": {
        "description": "Извлечение ключевой информации",
        "content": (
            "Из приведенного ниже текста вакансии извлеки ключевую информацию и представь ее в виде краткого структурированного резюме."
            "Требуемый формат ответа:\n"
            "Название должности:\n"
            "Уровень (грэйд):\n"
            "Ключевые обязанности (3-5 пунктов):\n"
            "Основные требования (3-5 пунктов):\n"
            "Условия (формат работы, з/п вилка, бенефиты):\n"
            "Текст вакансии:\n"
            "В быстрорастущий IT-продукт ищем опытного Python-разработчика (Middle+/Senior) для усиления команды бэкенда. Основной стек: Python, FastAPI, PostgreSQL, Redis, Docker, Kafka. В обязанности будет входить: разработка и поддержка микросервисной архитектуры нашего ядра продукта, оптимизация существующих API-эндпоинтов для увеличения скорости отклика, написание unit- и интеграционных тестов (pytest), участие в код-ревью. Мы ждем от кандидата уверенного влажения Python и асинхронным программированием, опыта работы с реляционными БД (желательно Postgres) от 3 лет, понимания принципов REST, опыта работы в команде по Agile/Scrum. Будет большим плюсом опыт с Kubernetes и любое exposure к ML-инструментам (scikit-learn). Мы предлагаем работу в гибридном формате (2 дня в неделю в современном офисе у м. Полянка) или полный remote, конкурентную заработную плату (вилка 180 000 - 250 000 руб. на руки после НДФЛ), ДМС, оплату обучения и конференций, современную технику (MacBook Pro M2)."
        )
    }
}

BASE_PARAMS = {
}

TUNED_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 500,
    "repeat_penalty": 1.2
}

def query_llm(
    model: str,
    prompt: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Отправляет запрос в Ollama и возвращает ответ + метрики.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **params
    }

    start_time = time.time()
    response = requests.post(OLLAMA_API_URL, json=payload)
    elapsed_time = time.time() - start_time

    response.raise_for_status()
    data = response.json()

    answer_text = data["choices"][0]["message"]["content"]

    return {
        "text": answer_text,
        "response_time_sec": elapsed_time,
        "raw_response": data
    }

def run_experiment():
    results = []

    for model_name, model_id in MODELS.items():
        for prompt_name, prompt_data in PROMPTS.items():
            for mode_name, params in [
                ("base", BASE_PARAMS),
                ("tuned", TUNED_PARAMS)
            ]:
                print(f"Модель={model_name}, Промпт={prompt_name}, Режим={mode_name}")

                result = query_llm(
                    model=model_id,
                    prompt=prompt_data["content"],
                    params=params
                )

                results.append({
                    "model": model_name,
                    "model_id": model_id,
                    "prompt_type": prompt_name,
                    "prompt_description": prompt_data["description"],
                    "mode": mode_name,
                    "generation_params": params,
                    "response_text": result["text"],
                    "response_length_chars": len(result["text"]),
                    "response_time_sec": result["response_time_sec"]
                })

    return results

if __name__ == "__main__":
    experiment_results = run_experiment()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"llm_experiment_results_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, ensure_ascii=False, indent=2)

    print(f"Эксперимент завершён. Результаты сохранены в {filename}")
