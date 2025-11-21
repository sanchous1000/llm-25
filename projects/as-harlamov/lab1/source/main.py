import json
import os
import time
from typing import Any, Dict

import requests


OLLAMA_API_URL = 'http://localhost:11434/v1/chat/completions'
MODELS = [
    'llama3.2:1b',
    'mistral:7b-instruct',
    'qwen:7b',
]

PROMPTS = {
    'P1_generation': (
        'Напиши вежливое письмо арендодателю с просьбой отремонтировать протекающий кран на кухне. '
        'Укажи, что проблема наблюдается уже неделю и мешает готовить.'
    ),
    'P2_classification': (
        'Классифицируй следующий запрос пользователя в одну из категорий: [\'Billing\', \'Tech support\', \'Sales\']. '
        'Текст: \'Мой счёт за прошлый месяц оказался вдвое выше обычного. Можете проверить, не ошиблись ли в расчётах?\'.\n'
        'В ответе приведи только категорию.'
    ),
    'P3_extraction': (
        'Извлеки из описания товара следующие поля: название, цена (в рублях), страна-производитель. '
        'Описание: \'Смартфон Galaxy S24 Ultra, 12/512 ГБ, цвет чёрный. Цена: 129 990 ₽. Произведено в Вьетнаме.\'\n'
        'В ответе приведи значения полей через запятую'
    ),
}

MODES = {
    'A_basic': {
        'temperature': 0.8,
        'num_predict': 128,
    },
    'B_tuned': {
        'temperature': 0.3,
        'repeat_penalty': 1.2,
        'num_predict': 256,
    },
}

RESULTS = []


def query_ollama(model: str, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'options': options,
    }
    start_time = time.time()
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            headers={'Authorization': 'Bearer unused', 'Content-Type': 'application/json'},
            timeout=120,
        )
        latency = (time.time() - start_time) * 1000  # ms
        if response.status_code != 200:
            return {'error': f'HTTP {response.status_code}', 'latency_ms': latency}
        data = response.json()
        return {
            'response': data['choices'][0]['message']['content'].strip(),
            'input_tokens': data.get('usage', {}).get('prompt_tokens', None),
            'output_tokens': data.get('usage', {}).get('completion_tokens', None),
            'latency_ms': latency,
            'model': model,
        }
    except Exception as e:
        return {
            'error': str(e),
            'latency_ms': (time.time() - start_time) * 1000,
        }


if __name__ == '__main__':
    for model in MODELS:
        print(f'\nТестируем модель: {model}')
        for prompt_name, prompt_text in PROMPTS.items():
            for mode_name, options in MODES.items():
                print(f'  → {prompt_name} | {mode_name}')
                result = query_ollama(model, prompt_text, options)
                record = {
                    'model': model,
                    'prompt': prompt_name,
                    'mode': mode_name,
                    'options_used': options,
                    'prompt_text': prompt_text,
                    **result
                }
                RESULTS.append(record)
    
    os.makedirs('../results', exist_ok=True)
    filename = '../results/ollama_experiments.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    
    print(f'\nВсе эксперименты завершены. Результаты сохранены в {filename}')
