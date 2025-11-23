import yaml
import json
import time
import requests
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"
PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"
RESULTS_PATH = Path(__file__).parent / "results.jsonl"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def call_ollama(model_id, prompt, params):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False,
        **params
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()

def main():
    config = load_yaml(CONFIG_PATH)
    prompts = load_yaml(PROMPTS_PATH)
    results_file = open(RESULTS_PATH, "a", encoding="utf-8")

    for model in config.get("models", []):
        model_id = model["model_id"]
        model_name = model.get("name", model_id)
        for prompt_entry in prompts.get("prompts", []):
            prompt_text = prompt_entry["text"]
            task_type = prompt_entry.get("type", "unknown")
            # Basic run
            basic_params = {}
            start = time.time()
            basic_res = call_ollama(model_id, prompt_text, basic_params)
            duration = time.time() - start
            results_file.write(json.dumps({
                "model": model_name,
                "model_id": model_id,
                "prompt": prompt_text,
                "type": task_type,
                "mode": "basic",
                "response": basic_res.get("response", ""),
                "duration_s": duration,
                "params": basic_params
            }) + "\n")
            # Tuned run – modify at least two hyper‑parameters
            tuned_params = {
                "temperature": 0.5,
                "top_p": 0.8,
                "max_tokens": 100
            }
            start = time.time()
            tuned_res = call_ollama(model_id, prompt_text, tuned_params)
            duration = time.time() - start
            results_file.write(json.dumps({
                "model": model_name,
                "model_id": model_id,
                "prompt": prompt_text,
                "type": task_type,
                "mode": "tuned",
                "response": tuned_res.get("response", ""),
                "duration_s": duration,
                "params": tuned_params
            }) + "\n")
    results_file.close()
    print(f"Results written to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
