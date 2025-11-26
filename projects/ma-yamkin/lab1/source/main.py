import json
import time
from openai import OpenAI
from datetime import datetime

OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY = "pass"

MODELS = [
    "mistral:7b-instruct-v0.3-q4_0",
    "llama3.1:8b-instruct-q4_0",
    "qwen2.5:3b"
]

PROMPTS = {
    "P1_generation": (
        "Напишите вежливое официальное письмо от имени компании «ТехноСфера» клиенту, "
        "в котором сообщается о переносе даты запуска нового сервиса с 15 на 22 апреля 2025 года. "
        "Поблагодарите за понимание и выразите уверенность в высоком качестве предоставляемого решения."
    ),
    "P2_classification": (
        'Классифицируйте следующее обращение клиента по одной из категорий: "Рыболовля", "Ресторанный бизнес" или "Продажа на маркетплейсах". '
        'Ответ дайте строго в виде одной метки без пояснений.\n\n'
        'Обращение: "Здравствуйте! Я хочу заказать стейк из свинины и воду."'
    ),
    "P3_extraction": (
        "Проанализируйте описание товара ниже. Выполните два действия:\n"
        "1. Извлеките следующие поля: Название, Производитель, Объём памяти (в ГБ), Тип (ноутбук/планшет/смартфон).\n"
        "2. Напишите краткое резюме (1–2 предложения) о ключевых особенностях устройства.\n\n"
        "Описание: \"Новинка 2025 года от Samsung — ультрабук Galaxy Book4 Edge с процессором Snapdragon X Elite, "
        "16 ГБ оперативной памяти, 512 ГБ SSD-накопителем и 14-дюймовым AMOLED-дисплеем. "
        "Идеален для работы и творчества благодаря автономности до 20 часов.\""
    )
}

MODES = {
    "basic": {"temperature": 0.8, "max_tokens": 128, "repeat_penalty": 1.1},
    "tuned": {"temperature": 0.3, "max_tokens": 256, "repeat_penalty": 1.3}
}

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)

def query_model_with_metrics(model: str, prompt: str, params: dict) -> dict:
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"repeat_penalty": params["repeat_penalty"]}
        )
        end_time = time.time()

        usage = response.usage
        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "total_time_sec": round(end_time - start_time, 2),
            "tokens_per_sec": round(usage.completion_tokens / (end_time - start_time), 1)
        }
    except Exception as e:
        return {
            "answer": f"[ОШИБКА: {str(e)}]",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_time_sec": 0,
            "tokens_per_sec": 0
        }

full_results = {}
timestamp = datetime.now().strftime("%H_%M_%S")

for model in MODELS:
    print(f"{model}")
    full_results[model] = {}
    for prompt_key, prompt_text in PROMPTS.items():
        print(prompt_key)
        full_results[model][prompt_key] = {}
        for mode_name, params in MODES.items():
            result = query_model_with_metrics(model, prompt_text, params)
            full_results[model][prompt_key][mode_name] = result

with open(f"llm_benchmark_results{timestamp}.json", "w", encoding="utf-8") as f:
    json.dump({
        "results": full_results
    }, f, ensure_ascii=False, indent=2)