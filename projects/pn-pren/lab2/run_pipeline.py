import subprocess
import sys


def run_step(name, cmd):
    print(name)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error in: {name}")
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full RAG pipeline')
    parser.add_argument('--skip-parse', action='store_true', help='Skip document parsing')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store')
    args = parser.parse_args()
    
    if not args.skip_parse:
        run_step("Parse documents", 
                 "python src/parse_docs.py --input data/docs --output output/markdown")
    
    run_step("Build index", "python src/build_index.py")
    
    rebuild_flag = "--rebuild" if args.rebuild else ""
    run_step("Load to Qdrant", f"python src/load_to_vector_store.py {rebuild_flag}")
    
    if not args.skip_eval:
        run_step("Evaluate", "python src/evaluate.py --queries data/eval_queries.json")
    
    print("pipeline completed")


if __name__ == '__main__':
    main()