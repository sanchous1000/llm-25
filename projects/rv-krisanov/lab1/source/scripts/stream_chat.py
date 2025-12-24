#!/usr/bin/env python3
import json
import asyncio
import httpx
import sys
import yaml
import click
import pandas as pd
import dtale

def load_prompt(prompts_yaml: str, name: str) -> str:
    with open(prompts_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    for prompt in data.get('prompts', []):
        if prompt.get('name') == name:
            return prompt.get('prompt', '')
    raise SystemExit(f"prompt '{name}' not found in {prompts_yaml}")


async def stream_chat( model: str, prompt: str, params: dict[str, int | float]) -> str:
    assistant_response = ""
    async with httpx.AsyncClient(timeout=None) as httpx_client:
        request_params = {
            "model": model,
            "messages": [{"role":"user","content":prompt}],
            "stream": True,
            **params
        }
        async with httpx_client.stream(
            "POST", f"http://{server_endpoint}/v1/chat/completions", json=request_params) as request:
            async for line in request.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if line.strip() == "data: [DONE]":
                    break
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                if delta:
                    assistant_response += delta
                    sys.stdout.write(delta)
                    sys.stdout.flush()
    print()
    return assistant_response


prompts_yaml = '../prompts.yaml'

p1 = load_prompt(prompts_yaml, 'P1')
p2 = load_prompt(prompts_yaml, 'P2')
p3 = load_prompt(prompts_yaml, 'P3')

models = [
    'mistral:7b',
    'qwen3:4b',
    'llama3.2:3b',
]

server_endpoint = '0.0.0.0:11434'

creative_params = {
    'temperature': 1.2,
    'max_tokens': 1500,
    'top_p': 0.95,
    'top_k': 60,
    'repetition_penalty': 0.9,
}

def main()->None:
    assistant_responses = []
    for model in models:
        for prompt, prompt_name in [(p1, 'P1'), (p2, 'P2'), (p3, 'P3')]:
            for params in [{}, creative_params]:
                assistant_responses.append({
                    'model': model,
                    'prompt': prompt,
                    'prompt_name': prompt_name,
                    'params': params,
                    'assistant_response': asyncio.run(stream_chat(model, prompt, params))
                })
            with open('assistant_responses.json', 'w') as f:
                json.dump(assistant_responses, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

