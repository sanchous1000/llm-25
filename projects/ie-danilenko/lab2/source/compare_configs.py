"""
Скрипт для сравнения разных конфигураций сплиттера, эмбеддингов и индекса.
Выполняет задание 5: сравнение минимум 2-3 вариантов конфигураций.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse


def run_build_index(config_file: str) -> bool:
    """Запускает build_index.py с указанной конфигурацией."""
    try:
        result = subprocess.run(
            [sys.executable, "build_index.py"],
            cwd=Path(__file__).parent,
            env={**os.environ, "CONFIG_FILE": config_file},
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Ошибка при построении индекса с конфигурацией {config_file}:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def run_load_to_vector_store(chunks_dir: str, index_dir: str, rebuild: bool = False) -> bool:
    """Запускает load_to_vector_store.py."""
    try:
        cmd = [
            sys.executable, "load_to_vector_store.py",
            "--chunks-dir", chunks_dir,
            "--index-dir", index_dir
        ]
        if rebuild:
            cmd.append("--rebuild")
        
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Ошибка при загрузке в векторное хранилище:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def run_evaluate_retrieval(faiss_index_dir: str, chunks_dir: str, output_file: str) -> bool:
    """Запускает evaluate_retrieval.py."""
    try:
        result = subprocess.run(
            [
                sys.executable, "evaluate_retrieval.py",
                "--faiss-index-dir", faiss_index_dir,
                "--chunks-dir", chunks_dir,
                "--output", output_file
            ],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Ошибка при оценке retrieval:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def load_evaluation_results(result_file: str) -> Dict[str, Any]:
    """Загружает результаты оценки."""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_configurations(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Сравнивает разные конфигурации.
    
    Args:
        configs: Список конфигураций для сравнения
    
    Returns:
        Словарь с результатами сравнения
    """
    
    comparison_results = {
        "configurations": [],
        "summary": {}
    }
    
    for i, config in enumerate(configs):
        config_name = config.get("name", f"config_{i+1}")
        print(f"\n{'='*60}")
        print(f"Обработка конфигурации: {config_name}")
        print(f"{'='*60}")
        
        # Сохраняем конфигурацию во временный файл
        config_file = Path(f"config_temp_{i}.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Строим индекс
        print(f"\n1. Построение индекса...")
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        target_config = config_dir / "config.json"
        with open(config_file, 'r') as src, open(target_config, 'w') as dst:
            dst.write(src.read())
        
        if not run_build_index(str(target_config)):
            print(f"Ошибка при построении индекса для {config_name}")
            continue
        
        # Загружаем в векторное хранилище
        chunks_dir = config.get("output_dir", "chunks")
        index_dir = f"faiss_index_{i}"
        print(f"\n2. Загрузка в векторное хранилище...")
        if not run_load_to_vector_store(chunks_dir, index_dir, rebuild=True):
            print(f"Ошибка при загрузке в векторное хранилище для {config_name}")
            continue
        
        # Оцениваем
        output_file = f"evaluation_{i}.json"
        print(f"\n3. Оценка качества retrieval...")
        if not run_evaluate_retrieval(index_dir, chunks_dir, output_file):
            print(f"Ошибка при оценке для {config_name}")
            continue
        
        # Загружаем результаты
        results = load_evaluation_results(output_file)
        comparison_results["configurations"].append({
            "name": config_name,
            "config": config,
            "results": results
        })
        
        # Удаляем временный файл
        config_file.unlink()
    
    # Формируем сводку сравнения
    if comparison_results["configurations"]:
        print(f"\n{'='*60}")
        print("Сравнение конфигураций")
        print(f"{'='*60}")
        
        # Сравниваем метрики
        summary = {}
        for config_result in comparison_results["configurations"]:
            name = config_result["name"]
            summary[name] = {}
            
            for eval_result in config_result["results"]["evaluations"]:
                search_type = eval_result["search_type"]
                if search_type not in summary[name]:
                    summary[name][search_type] = {}
                
                for metric_name, metric_value in eval_result["metrics"].items():
                    summary[name][search_type][metric_name] = metric_value["mean"]
        
        comparison_results["summary"] = summary
        
        # Выводим таблицу сравнения
        print("\nСравнительная таблица метрик:")
        print(f"{'Конфигурация':<20} {'Тип поиска':<10} ", end="")
        k_values = [5, 10]
        for k in k_values:
            print(f"Recall@{k:<8} Precision@{k:<8} ", end="")
        print("MRR")
        print("-" * 100)
        
        for config_result in comparison_results["configurations"]:
            name = config_result["name"]
            for eval_result in config_result["results"]["evaluations"]:
                search_type = eval_result["search_type"]
                print(f"{name:<20} {search_type:<10} ", end="")
                for k in k_values:
                    recall = eval_result["metrics"][f"Recall@{k}"]["mean"]
                    precision = eval_result["metrics"][f"Precision@{k}"]["mean"]
                    print(f"{recall:.4f}     {precision:.4f}     ", end="")
                mrr = eval_result["metrics"]["MRR"]["mean"]
                print(f"{mrr:.4f}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description='Сравнение конфигураций')
    parser.add_argument('--configs', type=str, nargs='+', required=True,
                       help='Пути к файлам конфигураций для сравнения')
    parser.add_argument('--output', type=str, default='config_comparison.json',
                       help='Файл для сохранения результатов сравнения')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Сравнение конфигураций")
    print("=" * 60)
    
    # Загружаем конфигурации
    configs = []
    for config_file in args.configs:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"Ошибка: файл конфигурации не найден: {config_file}")
            continue
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Извлекаем имя из имени файла
        config_name = config_path.stem
        config["name"] = config_name
        configs.append(config)
    
    if not configs:
        print("Ошибка: не загружено ни одной конфигурации")
        return
    
    print(f"\nЗагружено конфигураций для сравнения: {len(configs)}")
    
    # Сравниваем конфигурации
    comparison_results = compare_configurations(configs)
    
    # Сохраняем результаты
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Сравнение завершено!")
    print(f"Результаты сохранены в: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    import os
    main()

