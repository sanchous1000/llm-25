"""
Генерирует примеры ответов для двух вопросов с использованием двух версий индексов.

Создаёт answer_examples.json с ответами для демонстрации различий между версиями.
"""

import json
from pathlib import Path

from config_utils import load_config
from rag_pipeline import rag_query


def main():
    """Генерирует примеры ответов."""
    lab2_dir = Path(__file__).parent.parent
    
    # Два тестовых вопроса
    questions = [
        "Что такое FastAPI?",
        "Как работать с зависимостями в FastAPI?",
    ]
    
    # Две версии индексов (используем текущие имена из ES)
    experiments = [
        {
            "name": "baseline_recursive_1024",
            "version": "4530f7569b81",
            "index_name": "fastapi_docs_baseline_recursive_1024",
            "description": "Recursive splitter, chunk_size=1024, 371 chunks",
        },
        {
            "name": "markdown_512",
            "version": "e6233de3342d",
            "index_name": "fastapi_docs_markdown_512",
            "description": "Markdown splitter, chunk_size=512, 991 chunks",
        },
    ]
    
    # Загружаем базовый конфиг
    config_path = lab2_dir / "source/config.yaml"
    base_config = load_config(config_path)
    
    results = {
        "questions": questions,
        "experiments": [],
    }
    
    print("="*80)
    print("ГЕНЕРАЦИЯ ПРИМЕРОВ ОТВЕТОВ")
    print("="*80)
    
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"Эксперимент: {exp['name']}")
        print(f"Версия: {exp['version']}")
        print(f"Индекс: {exp['index_name']}")
        print(f"Описание: {exp['description']}")
        print(f"{'='*80}")
        
        exp_config = base_config.copy()
        if "elasticsearch" not in exp_config:
            exp_config["elasticsearch"] = {}
        exp_config["elasticsearch"]["index_name"] = exp["index_name"]
        
        exp_results = {
            "name": exp["name"],
            "version": exp["version"],
            "index_name": exp["index_name"],
            "description": exp["description"],
            "answers": [],
        }
        
        for i, question in enumerate(questions, 1):
            print(f"\nВопрос {i}: {question}")
            
            try:
                result = rag_query(question, exp_config)
                
                answer_data = {
                    "question": question,
                    "answer": result["answer"],
                    "sources": [
                        {
                            "index": src["index"],
                            "chunk_id": src.get("chunk_id", ""),
                            "source_path": src["source_path"],
                            "title": src.get("title", ""),
                            "header": src.get("header", ""),
                            "relevance_score": src["relevance_score"],
                            "snippet": src.get("snippet", "")[:150],
                        }
                        for src in result["sources"]
                    ],
                    "num_sources": result["num_sources"],
                }
                
                exp_results["answers"].append(answer_data)
                
                print(f"  ✓ Ответ получен (длина: {len(result['answer'])} символов)")
                print(f"    Источников: {result['num_sources']}")
                
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
                exp_results["answers"].append({
                    "question": question,
                    "answer": None,
                    "error": str(e),
                })
        
        results["experiments"].append(exp_results)
    
    # Сохраняем результаты
    output_file = lab2_dir / "data/evaluation/answer_examples.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("ГОТОВО!")
    print("="*80)
    print(f"Результаты сохранены в: {output_file}")
    print(f"\nВсего ответов: {sum(len(exp['answers']) for exp in results['experiments'])}")
    print("\nТеперь можно:")
    print("  1. Просмотреть answer_examples.json")
    print("  2. Сравнить качество ответов между версиями")
    print("  3. Проанализировать, какие источники использованы")


if __name__ == "__main__":
    main()
