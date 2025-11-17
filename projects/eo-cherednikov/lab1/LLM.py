import requests, time, json, os
from datetime import datetime

BASE_URL = "http://localhost:11434/v1/chat/completions"
OUTPUT_FILE = "results.jsonl"

models = ["qwen3-vl", "llama3.1", "mistral"]
prompts = {
    "P1": "Напиши вежливое, официальное письмо клиенту (ок. 120–200 слов) от имени службы поддержки интернет-магазина. Тон: вежливый, дружелюбный, конструктивный. Включи: 1) благодарность за обращение, 2) краткое объяснение причины задержки/проблемы, 3) конкретные шаги, которые мы предпринимаем, 4) контакты для обратной связи.",
    "P2": "Классифицируй сообщения в одну из меток: [\"Billing\",\"Tech support\",\"Sales\"]. Верни только название метки. Текст сообщения 1: \"Счет за прошлый месяц пришёл с ошибкой, сумма отличается.\" Текст сообщения 2: \"Не могу подключить роутер — индикатор не горит.\"",
    "P3": '''Дано описание товара — извлеки поля: {\"title\",\"price_estimate\",\"main_features\",\"short_summary\"}. Если цена неизвестна, укажи \"unknown\".  Описание: \"Держатель для миниатюр
            Надёжно фиксирует миниатюру при покраске, удобно лежит в руке благодаря эргономичной ручке
            В комплекте: ручка со слайдером, два зажима, одна резинка
            Механизм фиксации не делает прощальное "кря", если повернуть его сильно или не в ту сторону
            Под базы от 20 до 65 мм
            Материал: PETG-пластик (масло- и бензостойкий). Не боится солнечных лучей, в отличие от PLA
            Прочный, никаких слабых мест в конструкции
            Цвет изделия: чёрный, как на фото
            Обратите внимание: изготовлено послойным методом 3D-печати на профессиональном оборудовании (готовое изделие имеет слоистую текстуру)\"'''
}

mods = {
    "base": {},
    "tuned": {"temperature": 0.2, "max_tokens": 150, "top_p": 0.95, "repeat_penalty": 1.1}
}

def call_model(model, prompt, params):
    payload = {
        "model": model,
        "messages": [{"role":"user","content": prompt}],
    }
    payload.update(params)
    start = time.time()
    r = requests.post(BASE_URL, json=payload, timeout=120)
    elapsed = time.time() - start
    try:
        data = r.json()
    except Exception:
        data = {"error": r.text}
    return data, elapsed

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for model in models:
        for pid, prompt in prompts.items():
            for mode_name, params in mods.items():
                data, elapsed = call_model(model, prompt, params)
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "prompt_id": pid,
                    "prompt": prompt,
                    "mode": mode_name,
                    "params": params,
                    "elapsed_s": elapsed,
                    "response": data
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Wrote {model} {pid} {mode_name} ({elapsed:.2f}s)")
