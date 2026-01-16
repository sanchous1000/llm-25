import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


@dataclass
class ModelConfig:
    name: str
    model: str
    base_url: str
    api_key: str = "not-used"


@dataclass
class PromptSpec:
    pid: str
    task: str
    text: str


RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

# Три разнотипных промпта (генерация, классификация, суммаризация/извлечение).
PROMPTS: List[PromptSpec] = [
    PromptSpec(
        pid="P1_generation",
        task="Генерация пресс-релиза",
        text=(
            "Напиши вежливое письмо клиенту, который хочет впервые воспользоваться нашим "
            "ИИ-сервисом для генерации рекламных объявлений под Яндекс Директ. Подчеркни выгоды для малого бизнеса, "
            "установи доверительные отношения и предложи бесплатную демо-сессию."
        ),
    ),

    PromptSpec(
        pid="P2_classification",
        task="Классификация обращения",
        text=(
            "Проанализируй обращение клиента и классифицируй его одной категорией из списка "
            "['Billing','Tech support','Sales']. Ответ должен содержать только метку класса.\n\n"
            "Обращение клиента: «У меня не работает вход в личный кабинет, постоянно выдает ошибку 404. "
            "Помогите решить проблему.»"
        ),
    ),
    PromptSpec(
        pid="P3_summarization",
        task="Извлечение/суммаризация",
        text=(
            "Из описания товара извлеки название и ключевые характеристики и сделай краткое "
            "резюме в 2–3 предложениях. Описание: «Смарт-лампа LuxLight X2 "
            "поддерживает Wi‑Fi, регулируемую температуру цвета 2700–6500K, "
            "голосовое управление и расписания. Потребление 9 Вт, яркость 900 лм, "
            "цоколь E27, работает с Алиса/Google Home.»"
        ),
    ),
]

# Конфигурации моделей (можно править через переменные окружения).
MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        name="llama",
        model=os.getenv("LLAMA_MODEL", "llama3.1:8b-instruct-q4_0"),
        base_url=os.getenv("LLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLAMA_API_KEY", "not-used"),
    ),
    ModelConfig(
        name="mistral",
        model=os.getenv("MISTRAL_MODEL", "mistral:7b-instruct"),
        base_url=os.getenv("MISTRAL_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("MISTRAL_API_KEY", "not-used"),
    ),
    ModelConfig(
        name="qwen",
        model=os.getenv("QWEN_MODEL", "qwen2.5:3b"),
        base_url=os.getenv("QWEN_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("QWEN_API_KEY", "not-used"),
    ),
]

# Гиперпараметры для «тюнинга».
TUNED_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.8,
    "max_tokens": 400,
}


def build_client(cfg: ModelConfig) -> OpenAI:
    """Создаёт OpenAI-совместимый клиент для конкретного эндпоинта."""
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=300)


def run_completion(
    client: OpenAI,
    cfg: ModelConfig,
    prompt: PromptSpec,
    mode: str,
    tuned: bool,
) -> Dict[str, Any]:
    params = TUNED_PARAMS.copy() if tuned else {}
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt.text}],
        **params,
    )
    elapsed = time.perf_counter() - start
    message = resp.choices[0].message
    content = message.content if message else ""
    usage = getattr(resp, "usage", None)
    completion_tokens = (
        usage.completion_tokens if usage and usage.completion_tokens else None
    )
    prompt_tokens = usage.prompt_tokens if usage and usage.prompt_tokens else None
    total_tokens = usage.total_tokens if usage and usage.total_tokens else None

    # Если usage отсутствует, ставим нули, чтобы не падать.
    if completion_tokens is None:
        completion_tokens = 0
    if prompt_tokens is None:
        prompt_tokens = 0
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    # Вычисляем метрики производительности
    elapsed_rounded = round(elapsed, 3)
    generation_speed = (
        round(completion_tokens / elapsed, 2) if elapsed > 0 and completion_tokens > 0 else 0.0
    )
    total_throughput = (
        round(total_tokens / elapsed, 2) if elapsed > 0 and total_tokens > 0 else 0.0
    )
    time_per_completion_token = (
        round(elapsed / completion_tokens, 4) if completion_tokens > 0 else 0.0
    )

    return {
        "model_family": cfg.name,
        "model": cfg.model,
        "mode": mode,
        "prompt_id": prompt.pid,
        "prompt_task": prompt.task,
        "prompt_text": prompt.text,
        "hyperparameters": params,
        "response": content,
        "timing_sec": elapsed_rounded,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "performance": {
            "generation_speed_tokens_per_sec": generation_speed,
            "total_throughput_tokens_per_sec": total_throughput,
            "time_per_completion_token_sec": time_per_completion_token,
        },
    }


def aggregate_summary(results: List[Dict[str, Any]]) -> str:
    """
    Делает короткую аналитическую сводку: изменение длины ответа и времени
    между дефолтом и тюнингом для каждой связки модель × промпт.
    """
    lines: List[str] = []
    grouped: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for row in results:
        key = (row["model_family"], row["prompt_id"])
        grouped.setdefault(key, {})[row["mode"]] = row

    for (model_family, pid), modes in grouped.items():
        base = modes.get("baseline")
        tuned = modes.get("tuned")
        if not base or not tuned:
            continue
        dtokens = tuned["usage"]["completion_tokens"] - base["usage"]["completion_tokens"]
        dtime = tuned["timing_sec"] - base["timing_sec"]
        base_speed = base.get("performance", {}).get("generation_speed_tokens_per_sec", 0)
        tuned_speed = tuned.get("performance", {}).get("generation_speed_tokens_per_sec", 0)
        dspeed = tuned_speed - base_speed
        lines.append(
            f"{model_family} / {pid}: "
            f"токены Δ={dtokens:+}, время Δ={dtime:+.3f} c, "
            f"скорость Δ={dspeed:+.2f} токен/с "
            f"(base={base['timing_sec']} c, tuned={tuned['timing_sec']} c)"
        )
    if not lines:
        return "Недостаточно данных для сравнения."
    return "\n".join(lines)


def run_benchmark(model_configs: List[ModelConfig]) -> None:
    results: List[Dict[str, Any]] = []
    for cfg in model_configs:
        client = build_client(cfg)
        for prompt in PROMPTS:
            results.append(
                run_completion(client, cfg, prompt, mode="baseline", tuned=False)
            )
            results.append(
                run_completion(client, cfg, prompt, mode="tuned", tuned=True)
            )

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено {len(results)} прогонов в {RESULTS_PATH}")
    print("\nАналитика по влиянию гиперпараметров:")
    print(aggregate_summary(results))


if __name__ == "__main__":
    run_benchmark(MODEL_CONFIGS)