import argparse
import sys

from config import load_config


def run_fetch():
    from fetch_docs import fetch_all_docs
    config = load_config()
    fetch_all_docs(config)


def run_build(args):
    from build_index import build_index
    config = load_config(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        splitter_type=args.splitter,
        embedding_model=args.embedding_model
    )
    build_index(config, rebuild=args.rebuild)


def run_load(args):
    from load_to_vector_store import load_to_vector_store
    config = load_config(
        qdrant_host=args.host,
        qdrant_port=args.port
    )
    load_to_vector_store(config, rebuild=args.rebuild)


def run_evaluate(args):
    from evaluate import evaluate_retrieval, print_evaluation_report, save_evaluation_results
    config = load_config()
    results, summary = evaluate_retrieval(config, args.k)
    print_evaluation_report(summary, config)
    save_evaluation_results(results, summary, config)


def run_rag(args):
    from rag_pipeline import RAGPipeline
    config = load_config(
        top_k=args.top_k,
        llm_model=args.model
    )
    pipeline = RAGPipeline(config)
    
    if args.interactive:
        pipeline.interactive_mode()
    elif args.question:
        response = pipeline.query(args.question)
        print(f"\nQuestion: {response.question}")
        print(f"\nAnswer:\n{response.answer}")
        print("\nSources:")
        for src in response.sources:
            print(f"  - [{src['book']}] {src['section']} (score: {src['score']})")


def run_all(args):
    print("Step 1: Fetching documents...")
    run_fetch()
    
    print("\nStep 2: Building index...")
    build_args = argparse.Namespace(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        splitter=args.splitter,
        embedding_model=args.embedding_model,
        rebuild=True
    )
    run_build(build_args)
    
    print("\nStep 3: Loading to vector store...")
    load_args = argparse.Namespace(host="localhost", port=6333, rebuild=True)
    run_load(load_args)
    
    print("\nStep 4: Evaluating...")
    eval_args = argparse.Namespace(k=[5, 10])
    run_evaluate(eval_args)
    
    print("\nPipeline complete!")


def main():
    parser = argparse.ArgumentParser(description="YDKJS RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # fetch
    subparsers.add_parser("fetch", help="Fetch YDKJS documents from GitHub")
    
    # build
    build_parser = subparsers.add_parser("build", help="Build index from documents")
    build_parser.add_argument("--chunk-size", type=int, default=512)
    build_parser.add_argument("--chunk-overlap", type=int, default=50)
    build_parser.add_argument("--splitter", choices=["markdown", "recursive"], default="markdown")
    build_parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    build_parser.add_argument("--rebuild", action="store_true")
    
    # load
    load_parser = subparsers.add_parser("load", help="Load index to vector store")
    load_parser.add_argument("--host", default="localhost")
    load_parser.add_argument("--port", type=int, default=6333)
    load_parser.add_argument("--rebuild", action="store_true")
    
    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval quality")
    eval_parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    
    # rag
    rag_parser = subparsers.add_parser("rag", help="Run RAG pipeline")
    rag_parser.add_argument("--question", "-q", type=str)
    rag_parser.add_argument("--top-k", type=int, default=5)
    rag_parser.add_argument("--model", default="qwen2.5:3b")
    rag_parser.add_argument("--interactive", "-i", action="store_true")
    
    # all
    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument("--chunk-size", type=int, default=512)
    all_parser.add_argument("--chunk-overlap", type=int, default=50)
    all_parser.add_argument("--splitter", choices=["markdown", "recursive"], default="markdown")
    all_parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    
    args = parser.parse_args()
    
    if args.command == "fetch":
        run_fetch()
    elif args.command == "build":
        run_build(args)
    elif args.command == "load":
        run_load(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "rag":
        run_rag(args)
    elif args.command == "all":
        run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

