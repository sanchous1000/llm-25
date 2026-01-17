"""
Скрипт для проведения экспериментов с разными конфигурациями.

Запускает несколько экспериментов с разными параметрами чанкирования,
эмбеддингов и т.д., сохраняет результаты для сравнения.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config_utils import load_config, save_config


def run_command(cmd: list[str], description: str) -> bool:
    """Запускает команду."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(e.stdout)
        print(e.stderr)
        return False




def run_experiment(
    experiment_name: str,
    version: str,
    expected_chunks_file: str,
    base_config_path: Path,
    lab2_dir: Path,
) -> dict[str, Any] | None:
    """Запускает один эксперимент на основе существующей версии индекса.
    
    Returns:
        Результаты эксперимента или None при ошибке.
    """
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT: {experiment_name}")
    print(f"# Version: {version}")
    print(f"# Expected chunks: {expected_chunks_file}")
    print(f"{'#'*60}")
    
    script_dir = Path(__file__).parent
    
    # Проверяем существование версии
    version_dir = lab2_dir / "data" / "index" / version
    if not version_dir.exists():
        print(f"[ERROR] Version {version} not found in data/index/")
        return None
    
    # Загружаем metadata версии
    metadata_file = version_dir / "metadata.json"
    with open(metadata_file, "r", encoding="utf-8") as f:
        version_metadata = json.load(f)
    
    # Используем конфигурацию из metadata
    exp_config = version_metadata["config"].copy()
    
    # Обновляем expected_chunks_file в конфиге
    if "evaluation" not in exp_config:
        exp_config["evaluation"] = {}
    exp_config["evaluation"]["expected_chunks_file"] = expected_chunks_file
    
    # Создаем простое имя индекса для эксперимента (без дублирования)
    unique_index_name = f"fastapi_docs_{experiment_name}"
    if "elasticsearch" not in exp_config:
        exp_config["elasticsearch"] = {}
    exp_config["elasticsearch"]["index_name"] = unique_index_name
    
    # Создаем временный файл конфигурации
    exp_config_path = lab2_dir / f"source/config_experiment_{experiment_name}.yaml"
    save_config(exp_config, exp_config_path)
    
    results = {
        "name": experiment_name,
        "version": version,
        "expected_chunks_file": expected_chunks_file,
        "timestamp": datetime.now().isoformat(),
        "success": False,
    }
    
    try:
        # Индекс уже построен, версия известна
        results["version"] = version
        
        # Шаг 1: Загрузка в Elasticsearch из существующей версии
        if not run_command(
            [sys.executable, str(script_dir / "load_to_vector_store.py"), "--config", str(exp_config_path), "--version", version],
            f"Loading version {version} to Elasticsearch for {experiment_name}",
        ):
            return results
        
        # Сохраняем имя индекса
        index_name = exp_config.get("elasticsearch", {}).get("index_name")
        results["index_name"] = index_name
        
        # Шаг 2: Оценка
        if not run_command(
            [sys.executable, str(script_dir / "evaluate.py"), "--config", str(exp_config_path), "--version", version],
            f"Evaluating {experiment_name}",
        ):
            return results
        
        # Загружаем результаты
        results_file = lab2_dir / f"data/evaluation/results_{version}.json"
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                eval_results = json.load(f)
            
            results["metrics"] = eval_results["metrics"]
            results["success"] = True
            
            # Копируем results в experiment-specific файл
            exp_results_file = lab2_dir / f"data/evaluation/experiment_{experiment_name}.json"
            shutil.copy(results_file, exp_results_file)
            print(f"\n[OK] Results saved to: {exp_results_file}")
    
    finally:
        # Очищаем временный конфиг
        if exp_config_path.exists():
            exp_config_path.unlink()
    
    return results


def compare_experiments(results: list[dict[str, Any]], output_path: Path) -> None:
    """Сравнивает результаты экспериментов."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "experiments": results,
        "comparison": {},
    }
    
    # Собираем метрики для сравнения
    metrics_to_compare = ["recall@5", "precision@5", "mrr"]
    
    for metric_name in metrics_to_compare:
        comparison["comparison"][metric_name] = {}
        
        for exp in results:
            if exp["success"] and "metrics" in exp:
                mean_value = exp["metrics"].get(metric_name, {}).get("mean", 0)
                comparison["comparison"][metric_name][exp["name"]] = mean_value
    
    # Сохраняем сравнение
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    # Выводим таблицу сравнения
    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Experiment':<30} {'Recall@5':<12} {'Precision@5':<12} {'MRR':<12}")
    print('-' * 80)
    
    for exp in results:
        if not exp["success"]:
            print(f"{exp['name']:<30} FAILED")
            continue
        
        metrics = exp.get("metrics", {})
        recall = metrics.get("recall@5", {}).get("mean", 0)
        precision = metrics.get("precision@5", {}).get("mean", 0)
        mrr = metrics.get("mrr", {}).get("mean", 0)
        
        print(f"{exp['name']:<30} {recall:<12.4f} {precision:<12.4f} {mrr:<12.4f}")
    
    # Находим лучший эксперимент
    best_recall = max(
        (exp for exp in results if exp["success"]),
        key=lambda x: x["metrics"]["recall@5"]["mean"],
        default=None,
    )
    
    if best_recall:
        print(f"\n[BEST] Recall@5: {best_recall['name']} ({best_recall['metrics']['recall@5']['mean']:.4f})")
    
    print(f"\n[OK] Comparison saved to: {output_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with different configurations",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="source/config.yaml",
        help="Base configuration file",
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    lab2_dir = script_dir.parent
    base_config_path = lab2_dir / args.base_config
    
    if not base_config_path.exists():
        print(f"ERROR: Base config not found: {base_config_path}")
        return
    
    # Определяем эксперименты на основе существующих версий
    experiments = [
        {
            "name": "baseline_recursive_1024",
            "version": "4530f7569b81",
            "expected_chunks_file": "data/evaluation/expected_chunks_4530f7569b81.json",
        },
        {
            "name": "markdown_512",
            "version": "e6233de3342d",
            "expected_chunks_file": "data/evaluation/expected_chunks_e6233de3342d.json",
        },
    ]
    
    print("\n" + "="*80)
    print("STARTING EXPERIMENTS")
    print("="*80)
    print(f"Base config: {base_config_path}")
    print(f"Number of experiments: {len(experiments)}")
    print("Using pre-built indices from data/index/")
    
    # Запускаем эксперименты
    results = []
    for exp in experiments:
        result = run_experiment(
            exp["name"],
            exp["version"],
            exp["expected_chunks_file"],
            base_config_path,
            lab2_dir,
        )
        if result:
            results.append(result)
    
    # Сохраняем сравнение
    comparison_path = lab2_dir / "data/evaluation/experiments_comparison.json"
    compare_experiments(results, comparison_path)


if __name__ == "__main__":
    main()
