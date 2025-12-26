import json
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI


MODELS = [
    "qwen2.5:3b",
    "llama3.2:3b", 
    "mistral:7b"
]

PROMPTS = {
    "generation": {
        "name": "Генерация текста",
        "prompt": """Напиши короткое вежливое письмо коллеге с просьбой перенести встречу с понедельника на среду. 
Причина — срочная командировка. Письмо должно быть формальным, но дружелюбным."""
    },
    "classification": {
        "name": "Классификация",
        "prompt": """Определи категорию следующего обращения клиента. 
Возможные категории: ["Billing", "Tech support", "Sales"]
Ответь только названием категории на английском.

Обращение: "Добрый день! Вчера оплатил подписку, но деньги списались дважды. Прошу разобраться и вернуть лишнее списание."
"""
    },
    "extraction": {
        "name": "Извлечение информации",
        "prompt": """Извлеки структурированную информацию из описания товара. 
Верни результат в формате JSON с полями: название, цена, цвет, размер, материал.

Описание: "Кроссовки Nike Air Max 90, белого цвета, размер 42, верх из натуральной кожи и текстиля. Цена 12990 рублей."
"""
    }
}

BASE_PARAMS = {}

TUNED_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.85
}


def create_client():
    return OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )


def run_inference(client, model, prompt, params=None):
    start_time = time.time()
    
    request_params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    if params:
        request_params.update(params)
    
    try:
        response = client.chat.completions.create(**request_params)
        elapsed = time.time() - start_time
        
        output_text = response.choices[0].message.content
        usage = response.usage
        
        return {
            "output": output_text,
            "input_tokens": usage.prompt_tokens if usage else None,
            "output_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
            "time_seconds": round(elapsed, 2),
            "tokens_per_second": round(usage.completion_tokens / elapsed, 2) if usage and elapsed > 0 else None
        }
    except Exception as e:
        return {
            "error": str(e),
            "output": None
        }


def main():
    client = create_client()
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": MODELS,
        "prompts": {k: v["name"] for k, v in PROMPTS.items()},
        "tuned_params": TUNED_PARAMS,
        "runs": []
    }
    
    for model in MODELS:
        print(f"Модель: {model}")

        for prompt_type, prompt_data in PROMPTS.items():
            print(f"\n--- {prompt_data['name']} ---")
            
            # Base run
            print("base запуск...")
            base_result = run_inference(client, model, prompt_data["prompt"], BASE_PARAMS)
            
            # Tuned run
            print("tune запуск...")
            tuned_result = run_inference(client, model, prompt_data["prompt"], TUNED_PARAMS)
            
            run_entry = {
                "model": model,
                "prompt_type": prompt_type,
                "prompt_name": prompt_data["name"],
                "prompt_text": prompt_data["prompt"],
                "base": base_result,
                "tuned": tuned_result
            }
            results["runs"].append(run_entry)
            
            print(f"  base:  {base_result.get('time_seconds', 'N/A')}s, "
                  f"{base_result.get('output_tokens', 'N/A')} tokens")
            print(f"  tuned: {tuned_result.get('time_seconds', 'N/A')}s, "
                  f"{tuned_result.get('output_tokens', 'N/A')} tokens")
    
    output_path = Path(__file__).parent.parent / "results" / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nРезультаты сохранены в {output_path}")
    
    print_summary(results)


def print_summary(results):
    print("Results: ")

    for run in results["runs"]:
        print(f"\n[{run['model']}] {run['prompt_name']}")

        if run["base"].get("output"):
            print(f"Base:  {run['base']['output_tokens']} tok, {run['base']['time_seconds']}s")
            print(f"       {run['base']['output'][:100]}...")
        else:
            print(f"Base:  ОШИБКА - {run['base'].get('error', 'unknown')}")
        
        if run["tuned"].get("output"):
            print(f"Tuned: {run['tuned']['output_tokens']} tok, {run['tuned']['time_seconds']}s")
            print(f"       {run['tuned']['output'][:100]}...")
        else:
            print(f"Tuned: ОШИБКА - {run['tuned'].get('error', 'unknown')}")


if __name__ == "__main__":
    main()

