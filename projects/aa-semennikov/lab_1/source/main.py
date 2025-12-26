import json
import time
from openai import OpenAI
from datetime import datetime

OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY = "pass"

MODELS = [
    "qwen2.5:0.5b", 
    "llama3.1:8b",
    "mistral:7b"
]

def load_prompts():
    with open("./source/prompts/generation.txt", "r", encoding="utf-8") as f:
        generation = f.read().strip()
    with open("./source/prompts/classification.txt", "r", encoding="utf-8") as f:
        classification = f.read().strip()
    with open("./source/prompts/summarization.txt", "r", encoding="utf-8") as f:
        summarization = f.read().strip()
    
    return {
        "P1_generation": generation,
        "P2_classification": classification,
        "P3_summarization": summarization
    }

PROMPTS = load_prompts()

MODES = {
    "basic": {"temperature": 0.8, "max_tokens": 128, "repeat_penalty": 1.1},
    "tuned": {"temperature": 1.5, "max_tokens": 256, "repeat_penalty": 1.3}
}

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)

def query_model(model: str, prompt: str, params: dict) -> dict:
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
            "total_time_sec": round(end_time - start_time, 2)
        }
    except Exception as e:
        return {
            "answer": f"[ОШИБКА: {str(e)}]",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_time_sec": 0
        }

results = {}

for model in MODELS:
    print(f"Тестируется модель: {model}")
    results[model] = {}
    for prompt_key, prompt_text in PROMPTS.items():
        print(f"Текущий промпт: {prompt_key}")
        results[model][prompt_key] = {}
        for mode_name, params in MODES.items():
            result = query_model(model, prompt_text, params)
            results[model][prompt_key][mode_name] = result

with open(f"results.json", "w", encoding="utf-8") as f:
    json.dump({
        "results": results
    }, f, ensure_ascii=False, indent=2)