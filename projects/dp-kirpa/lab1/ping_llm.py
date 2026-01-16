import time
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)

models = ["qwen2.5:1.5b", "llama3", "mistral"]

prompts = {
    "P1 (Генерация)": "Напиши вежливое письмо клиенту о том, что доставка его заказа задерживается на 3 дня из-за плохой погоды. Предложи скидку 5% на следующий заказ.",
    "P2 (Классификация)": """Определи категорию запроса. Выбери только одну метку из списка: ["Billing", "Tech support", "Sales"].
    Текст запроса: 'У меня списали деньги дважды за один месяц, верните средства.'
    Ответ (только метка):""",
    "P3 (Извлечение)": """Извлеки данные о товаре в формате JSON (ключи: name, price, color).
    Текст: 'Продаются отличные кроссовки Nike Air, цвет красный, всего за 15000 рублей.'
    Ответ (только JSON):"""
}

configs = {
    "Default": {},
    "Tuned": {
        "temperature": 0.1,
        "max_tokens": 200,
        "presence_penalty": 1.1,
    }
}

results = []

print(f"{'Model':<10} | {'Mode':<8} | {'Task':<20} | {'Time (s)':<8} | {'Tokens out'}")
print("-" * 70)

for model in models:
    for task_name, prompt_text in prompts.items():
        for config_name, params in configs.items():
            
            start_time = time.time()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_text}],
                    **params
                )
                
                duration = time.time() - start_time
                content = response.choices[0].message.content
                out_tokens = response.usage.completion_tokens if response.usage else len(content)/3
                
                print(f"{model:<10} | {config_name:<8} | {task_name:<20} | {duration:.2f}     | {out_tokens}")
                
                with open("results_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"=== {model} | {config_name} | {task_name} ===\n")
                    f.write(f"Params: {params}\n")
                    f.write(f"Time: {duration:.2f}s\n")
                    f.write(f"OUTPUT:\n{content}\n")
                    f.write("-" * 40 + "\n")
                    
            except Exception as e:
                print(f"Error with {model}: {e}")
