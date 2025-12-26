"""
Script to compare multiple RAG configurations and evaluate their performance.
This allows testing different chunk sizes, overlap, splitter types, and vectorization methods.
"""
import yaml
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from build_index import build_index
from evaluate import evaluate_retrieval, load_evaluation_questions

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
EVAL_QUESTIONS_PATH = Path(__file__).parent.parent / "evaluation_questions.json"

def save_config(config: Dict, path: Path):
    """Save configuration to YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def load_config_from_file(path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compare_configurations(configs: List[Dict[str, Any]], rebuild: bool = True):
    """
    Compare multiple configurations by building indices and evaluating each.
    
    Args:
        configs: List of configuration dictionaries to test
        rebuild: Whether to rebuild index for each configuration
    """
    original_config = load_config_from_file(CONFIG_PATH)
    results = []
    
    print("="*80)
    print("RAG CONFIGURATION COMPARISON")
    print("="*80)
    print(f"Testing {len(configs)} configurations...\n")
    
    for i, config in enumerate(configs, 1):
        config_name = config.get("name", f"Config {i}")
        print(f"\n{'='*80}")
        print(f"Configuration {i}/{len(configs)}: {config_name}")
        print(f"{'='*80}")
        
        # Display configuration
        print("\nConfiguration:")
        print(f"  Splitter type: {config.get('splitter', {}).get('type', 'N/A')}")
        print(f"  Chunk size: {config.get('data', {}).get('chunk_size', 'N/A')}")
        print(f"  Chunk overlap: {config.get('data', {}).get('chunk_overlap', 'N/A')}")
        print(f"  Include headers: {config.get('splitter', {}).get('include_headers', 'N/A')}")
        print(f"  Vectorization: {config.get('vectorization', {}).get('type', 'N/A')}")
        print(f"  Embedding model: {config.get('embedding', {}).get('model_name', 'N/A')}")
        
        # Save configuration temporarily
        save_config(config, CONFIG_PATH)
        
        try:
            # Build index
            if rebuild:
                print("\nBuilding index...")
                build_index(rebuild=True)
            else:
                print("\nUsing existing index (skipping rebuild)...")
            
            # Evaluate
            print("\nEvaluating...")
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from rag_pipeline import RAGAgent
            agent = RAGAgent()
            questions = load_evaluation_questions()
            
            eval_results = evaluate_retrieval(agent, questions, k_values=[5, 10])
            
            # Store results
            result = {
                "config_name": config_name,
                "configuration": config,
                "metrics": eval_results["summary"],
                "detailed_results": eval_results["detailed_results"]
            }
            results.append(result)
            
            # Print summary
            print(f"\nResults for {config_name}:")
            metrics = eval_results["summary"]
            for k in [5, 10]:
                print(f"  Recall@{k}: {metrics[f'avg_recall@{k}']:.3f}")
                print(f"  Precision@{k}: {metrics[f'avg_precision@{k}']:.3f}")
            print(f"  MRR: {metrics['avg_mrr']:.3f}")
            
        except Exception as e:
            print(f"\nERROR evaluating {config_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "config_name": config_name,
                "configuration": config,
                "error": str(e)
            })
    
    # Restore original configuration
    save_config(original_config, CONFIG_PATH)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by MRR (descending)
    valid_results = [r for r in results if "metrics" in r]
    valid_results.sort(key=lambda x: x["metrics"]["avg_mrr"], reverse=True)
    
    print("\nRanked by MRR (Mean Reciprocal Rank):")
    print("-" * 80)
    for i, result in enumerate(valid_results, 1):
        metrics = result["metrics"]
        print(f"\n{i}. {result['config_name']}")
        print(f"   Recall@5: {metrics['avg_recall@5']:.3f} | "
              f"Precision@5: {metrics['avg_precision@5']:.3f} | "
              f"MRR: {metrics['avg_mrr']:.3f}")
        print(f"   Recall@10: {metrics['avg_recall@10']:.3f} | "
              f"Precision@10: {metrics['avg_precision@10']:.3f}")
    
    # Save comparison results
    output_path = Path(__file__).parent.parent / "configuration_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nDetailed comparison results saved to: {output_path}")
    
    return results

def create_default_configurations() -> List[Dict[str, Any]]:
    """Create a set of default configurations to compare."""
    base_config = load_config_from_file(CONFIG_PATH)
    
    configs = [
        {
            "name": "Baseline - Recursive Splitter, Sparse",
            **base_config,
            "splitter": {"type": "recursive", "include_headers": True},
            "data": {"source_dir": "data", "chunk_size": 500, "chunk_overlap": 50},
            "vectorization": {"type": "sparse"}
        },
        {
            "name": "Recursive Splitter, Dense",
            **base_config,
            "splitter": {"type": "recursive", "include_headers": True},
            "data": {"source_dir": "data", "chunk_size": 500, "chunk_overlap": 50},
            "vectorization": {"type": "dense"}
        },
        {
            "name": "Recursive Splitter, Hybrid",
            **base_config,
            "splitter": {"type": "recursive", "include_headers": True},
            "data": {"source_dir": "data", "chunk_size": 500, "chunk_overlap": 50},
            "vectorization": {"type": "hybrid"}
        },
        {
            "name": "Small Chunks (300), Dense",
            **base_config,
            "splitter": {"type": "recursive", "include_headers": True},
            "data": {"source_dir": "data", "chunk_size": 300, "chunk_overlap": 30},
            "vectorization": {"type": "dense"}
        },
        {
            "name": "Large Chunks (800), Dense",
            **base_config,
            "splitter": {"type": "recursive", "include_headers": True},
            "data": {"source_dir": "data", "chunk_size": 800, "chunk_overlap": 80},
            "vectorization": {"type": "dense"}
        },
        {
            "name": "Token Splitter, Dense",
            **base_config,
            "splitter": {"type": "token", "include_headers": True},
            "data": {"source_dir": "data", "chunk_size": 500, "chunk_overlap": 50},
            "vectorization": {"type": "dense"}
        },
        {
            "name": "Markdown Only Splitter, Dense",
            **base_config,
            "splitter": {"type": "markdown_only", "include_headers": True},
            "data": {"source_dir": "data", "chunk_size": 500, "chunk_overlap": 50},
            "vectorization": {"type": "dense"}
        }
    ]
    
    return configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple RAG configurations")
    parser.add_argument("--configs", type=str, help="Path to JSON file with configurations to test")
    parser.add_argument("--no-rebuild", action="store_true", help="Don't rebuild index for each config (use existing)")
    parser.add_argument("--default", action="store_true", help="Use default set of configurations")
    args = parser.parse_args()
    
    if args.configs:
        # Load configurations from file
        with open(args.configs, "r", encoding="utf-8") as f:
            configs_data = json.load(f)
            configs = configs_data.get("configurations", [])
    elif args.default:
        # Use default configurations
        configs = create_default_configurations()
    else:
        # Use default configurations
        print("No configuration file specified. Using default configurations.")
        print("Use --configs <file.json> to specify custom configurations or --default to explicitly use defaults.")
        configs = create_default_configurations()
    
    compare_configurations(configs, rebuild=not args.no_rebuild)

