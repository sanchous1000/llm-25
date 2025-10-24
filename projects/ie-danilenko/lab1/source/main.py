import json
import time
import os
from openai import OpenAI
from tabulate import tabulate

# Модели для тестирования
MODELS = [
    {"name": "qwen2.5:7b", "family": "Qwen"},
    {"name": "llama3.1:8b", "family": "Llama"},
    {"name": "mistral:7b", "family": "Mistral"}
]

def load_prompts():
    prompts = {}
    prompts_dir = "prompts"
    
    for filename in ["generation.txt", "classification.txt", "extraction.txt"]:
        prompt_name = filename.replace(".txt", "")
        filepath = os.path.join(prompts_dir, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                prompts[prompt_name] = f.read().strip()
        except FileNotFoundError:
            print(f"⚠️ Файл {filepath} не найден")
            prompts[prompt_name] = f"Промпт {prompt_name} не загружен"
    
    return prompts

def test_model(model_name, prompt_text, params):
    try:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            extra_body={"repeat_penalty": params["repeat_penalty"]}
        )
        generation_time = time.time() - start_time
        
        return {
            "model": model_name,
            "response": response.choices[0].message.content,
            "time": generation_time,
            "tokens": response.usage.completion_tokens if response.usage else 0,
            "params": params,
            "family": "Unknown"
        }
    except Exception as e:
        print(f"Ошибка с {model_name}: {e}")
        return None

def run_tests():
    print("Запуск тестов LLM моделей")
    print("=" * 40)
    
    prompts = load_prompts()
    
    results = []
    
    based_params = {"temperature": 1.1, "max_tokens": 256, "repeat_penalty": 1.1}
    
    tuned_params = {"temperature": 0.5, "max_tokens": 512, "repeat_penalty": 1.3}
    
    for model_info in MODELS:
        print(f"\nТестирование: {model_info['name']} ({model_info['family']})")
        
        for prompt_key, prompt_text in prompts.items():
            print(f"  {prompt_key}: ")
            
            # basic тест
            result = test_model(model_info["name"], prompt_text, based_params)
            if result:
                result["prompt"] = prompt_key
                result["mode"] = "basic"
                result["family"] = model_info["family"]
                results.append(result)
                print(f"    [OK] basic ({result['time']:.1f}с, {result['tokens']} токенов)")
            else:
                print("    [FAIL] basic")
            
            # tuned тест
            result = test_model(model_info["name"], prompt_text, tuned_params)
            if result:
                result["prompt"] = prompt_key
                result["mode"] = "tuned"
                result["family"] = model_info["family"]
                results.append(result)
                print(f"    [OK] tuned ({result['time']:.1f}с, {result['tokens']} токенов)")
            else:
                print("    [FAIL] tuned")
    
    return results

def analyze_results(results):
    print("\nАнализ результатов")
    print("=" * 50)
    
    # Статистика по моделям
    model_stats = []
    for model_name in set(r["model"] for r in results):
        model_results = [r for r in results if r["model"] == model_name]
        avg_time = sum(r["time"] for r in model_results) / len(model_results)
        avg_tokens = sum(r["tokens"] for r in model_results) / len(model_results)
        avg_speed = avg_tokens / avg_time if avg_time > 0 else 0
        
        model_stats.append([
            model_name,
            f"{avg_time:.1f}с",
            f"{avg_tokens:.0f}",
            f"{avg_speed:.1f} ток/с"
        ])
    
    print("\nСтатистика по моделям:")
    print(tabulate(model_stats, 
                   headers=["Модель", "Среднее время", "Средние токены", "Скорость"],
                   tablefmt="grid"))
    
    # Сравнение режимов
    basic_results = [r for r in results if r["mode"] == "basic"]
    tuned_results = [r for r in results if r["mode"] == "tuned"]
    
    if basic_results and tuned_results:
        basic_avg_time = sum(r["time"] for r in basic_results) / len(basic_results)
        tuned_avg_time = sum(r["time"] for r in tuned_results) / len(tuned_results)
        basic_avg_tokens = sum(r["tokens"] for r in basic_results) / len(basic_results)
        tuned_avg_tokens = sum(r["tokens"] for r in tuned_results) / len(tuned_results)
        basic_avg_speed = basic_avg_tokens / basic_avg_time if basic_avg_time > 0 else 0
        tuned_avg_speed = tuned_avg_tokens / tuned_avg_time if tuned_avg_time > 0 else 0
        
        comparison_data = [
            ["Время (с)", f"{basic_avg_time:.1f}", f"{tuned_avg_time:.1f}", f"{tuned_avg_time - basic_avg_time:+.1f}"],
            ["Токены", f"{basic_avg_tokens:.0f}", f"{tuned_avg_tokens:.0f}", f"{tuned_avg_tokens - basic_avg_tokens:+.0f}"],
            ["Скорость (ток/с)", f"{basic_avg_speed:.1f}", f"{tuned_avg_speed:.1f}", f"{tuned_avg_speed - basic_avg_speed:+.1f}"]
        ]
        
        print("\nСравнение режимов:")
        print(tabulate(comparison_data,
                       headers=["Метрика", "Basic", "Tuned", "Разница"],
                       tablefmt="grid"))
    
    # Статистика по типам промптов
    prompt_stats = []
    for prompt_type in set(r["prompt"] for r in results):
        prompt_results = [r for r in results if r["prompt"] == prompt_type]
        avg_time = sum(r["time"] for r in prompt_results) / len(prompt_results)
        avg_tokens = sum(r["tokens"] for r in prompt_results) / len(prompt_results)
        avg_speed = avg_tokens / avg_time if avg_time > 0 else 0
        
        prompt_name = {
            "generation": "P1: Генерация",
            "classification": "P2: Классификация", 
            "extraction": "P3: Извлечение"
        }.get(prompt_type, prompt_type)
        
        prompt_stats.append([
            prompt_name,
            f"{avg_time:.1f}с",
            f"{avg_tokens:.0f}",
            f"{avg_speed:.1f} ток/с"
        ])
    
    print("\nСтатистика по типам промптов:")
    print(tabulate(prompt_stats,
                   headers=["Тип промпта", "Среднее время", "Средние токены", "Скорость"],
                   tablefmt="grid"))

def save_results(results):
    print("\nСохранение результатов...")
    
    os.makedirs("results", exist_ok=True)
    
    # Сохранение сырых данных
    with open("results/raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Создание единой таблицы сравнения
    with open("results/comparison_table.md", "w", encoding="utf-8") as f:
        f.write("# Таблица со сравнением результатов моделей\n\n")
        f.write("| Задача | Модель | Вариант | Время (сек) | Выходные токены | Скорость (токенов/сек) | Ответ |\n")
        f.write("|--------|--------|---------|-------------|-----------------|----------------------|-------|\n")
        
        for result in results:
            # Вычисляем скорость генерации
            speed = result["tokens"] / result["time"] if result["time"] > 0 else 0
            
            # Форматируем ответ для Markdown таблицы
            response_text = result["response"].replace("\n", "<br/>").replace("|", "\\|")
            
            # Определяем задачу по промпту
            task_map = {
                "generation": "P1_generation",
                "classification": "P2_classification", 
                "extraction": "P3_extraction"
            }
            task = task_map.get(result["prompt"], result["prompt"])
            
            f.write(f"| {task} | {result['model']} | {result['mode']} | {result['time']:.2f} | {result['tokens']} | {speed:.1f} | {response_text} |\n")
    
    print("Результаты сохранены в директории results/")
    print("Таблица сравнения: results/comparison_table.md")

if __name__ == "__main__":
    results = run_tests()
    
    analyze_results(results)
    
    save_results(results)
    
    print("\nАнализ завершен!")
