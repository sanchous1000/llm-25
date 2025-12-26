import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Union

import pandas as pd
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_HF_NAMES = {
    "qwen2.5": "Qwen_Qwen2.5-7B-Instruct",
    "ministral": "mistralai_Ministral-8B-Instruct-2410",
    "yandex": "yandex_YandexGPT-5-Lite-8B-instruct",
    "gemma": "google_gemma-3-4b-it"
}

SAMPLING_PARAMS = {
    "temperature": 0.5,
    "top_p": 0.6,
    "extra_body": {"repetition_penalty": 1.1},
}

PROMPTS = {
    "P1": [
        {
            "role": "system",
            "content": "Твоя задача быть вежливым ассистентом компании AZOT, твое имя Боб, укажи что ты ИИ ассистент",
        },
        {
            "role": "user",
            "content": "Напиши письмо нашему клиенту Стиву, о том, что его доставка AZOT, а именно заказ номер 123321"
                       "задерживается в связи с (придумай необычную причину) на 2 года. К сожалению он не сможет вернуть заказ, так как (придумай почему)."
                       "Но скажи не расстраиваться и предложи купон на скидку 100 руб на заказ от 10 тыс. руб.",
        },
    ],
    "P2": [
        {
            "role": "system",
            "content": "Ты менеджер в системе поддержки службы доставки. Классифицируй сообщение от клиента по списку меток: ['Оплата', 'Доставка', 'Возврат', 'Аккаунт', 'Скидки', 'Прочее']."
                       "Вернуть необходимо одну метку в формате JSON, например: '{\"class\": \"Оплата\"}'",
        },
        {
            "role": "user",
            "content": "Здравствуйте. Вчера я делал заказ, где указывал оплату карточкой. Помимо того, что все привезли холодным, еще и помятым."
                       "Что случилось? Почему так плохо стало? Я в тот же день просил промокод, но ответа не получил. Пишу во второй раз.",
        },
    ],
    "P3": [
        {
            "role": "system",
            "content": "Твоя задача извлекать информацию для карточки товара из его описания. Отвечай в формате: - **Атрибут1**: Значение1\n - **Атрибут2**: Значение2",
        },
        {
            "role": "user",
            "content": "Новый смартфон iPhone 16 Pro Max оснащен дисплеем Super Retina XDR диагональю 6,9 дюйма с частотой обновления 120 Гц и разрешением 2796×1290 пикселей. Корпус выполнен из титана пятого класса и доступен в четырех цветах: серебристый, графитовый, синий и бежевый. Процессор A18 Pro с 6‑ядерным GPU обеспечивает прирост производительности на 20% по сравнению с предыдущим поколением. Объем оперативной памяти составляет 8 ГБ, встроенной памяти — 256, 512 ГБ или 1 ТБ. Основная камера на 48 Мп поддерживает съемку видео в 4K при 60 кадрах в секунду, фронтальная — 12 Мп с улучшенной стабилизацией. Аккумулятор ёмкостью 4500 мА·ч обеспечивает до 29 часов воспроизведения видео. Поддерживается быстрая зарядка мощностью до 30 Вт и беспроводная — до 15 Вт через MagSafe. Смартфон защищен по стандарту IP68 и способен выдерживать погружение в воду на глубину до 6 метров в течение 30 минут.Рекомендованная цена на старте продаж — от 1399 долларов США. Старт продаж в России запланирован на 27 сентября 2025 года.",
        },
    ],
}

BASE_URL = "http://localhost:8000/v1"


@dataclass
class FinishedRun:
    prompt_id: Optional[str]
    is_tuned: bool
    model: str
    temperature: Optional[Union[str, float]]
    top_p: Optional[Union[str, float]]
    repetition_penalty: Optional[Union[str, float]]
    duration: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    tts: Optional[float]
    output_text: str


def run(client: OpenAI, model: str, messages, params: Optional[dict] = None):
    params = params or {}
    body = {
        "model": model,
        "messages": messages,
        **params,
    }

    t0 = time.perf_counter()
    resp = client.chat.completions.create(**body)
    t1 = time.perf_counter()

    text = ""
    if resp.choices and resp.choices[0].message:
        text = resp.choices[0].message.content or ""

    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return FinishedRun(
        prompt_id="",
        is_tuned=bool(params),
        model=model,
        temperature=params.get("temperature", "1.0"),
        top_p=params.get("top_p", "1.0"),
        repetition_penalty=params.get("extra_body", {}).get(
            "repetition_penalty", "1.0"
        ),
        duration=t1 - t0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tts=completion_tokens / (t1 - t0),
        output_text=text,
    )


def bench() -> pd.DataFrame:
    rows = []

    for prompt, message in PROMPTS.items():
        logger.info(f"{prompt} with default params")
        rec_a = run(client, MODEL, message)
        logger.info(f"{prompt}: COMPLETED in {rec_a.duration:.2f} seconds")
        rec_a.prompt_id = prompt
        rows.append(asdict(rec_a))

        logger.info(f"{prompt} with custom params")
        rec_b = run(client, MODEL, message, SAMPLING_PARAMS)
        logger.info(f"{prompt}: COMPLETED in {rec_b.duration:.2f} seconds")
        rec_b.prompt_id = prompt
        rows.append(asdict(rec_b))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    client = OpenAI(base_url=BASE_URL, api_key="amongus")
    MODEL = client.models.list().data[0].id
    logger.info(f"MODEL: {MODEL}")

    # Небольшой прогрев
    for i in range(2):
        bench()

    results = bench()
    os.makedirs("results", exist_ok=True)
    results.to_csv(f"results/{MODEL_HF_NAMES[MODEL]}.csv")
    logger.info(f"Results saved to {MODEL_HF_NAMES[MODEL]}.csv")
