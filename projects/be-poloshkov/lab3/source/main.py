import argparse

from config import load_config


def run_demo(args):
    from rag_traced import TracedRAGPipeline
    
    config = load_config()
    pipeline = TracedRAGPipeline(config)
    
    if args.question:
        result = pipeline.query(
            args.question, 
            user_id=args.user_id, 
            session_id=args.session_id
        )
        print(f"\nQuestion: {result.question}")
        print(f"\nAnswer:\n{result.answer}")
        print(f"\nSources:")
        for src in result.sources:
            print(f"  - [{src['book']}] {src['section']} (score: {src['score']})")
        print(f"\nTimings: retrieval={result.retrieval_time:.2f}s, generation={result.generation_time:.2f}s")
    else:
        from rag_traced import demo
        demo()
    
    pipeline.flush()


def run_create_dataset(args):
    from dataset import create_dataset
    config = load_config()
    create_dataset(config)


def run_experiment(args):
    from experiment import ExperimentRunner
    
    config = load_config()
    runner = ExperimentRunner(config)
    runner.run_experiment(args.name, top_k=args.top_k)


def run_comparison(args):
    from experiment import run_comparison_experiments
    config = load_config()
    run_comparison_experiments(config)


def run_interactive(args):
    from rag_traced import TracedRAGPipeline
    
    config = load_config()
    pipeline = TracedRAGPipeline(config)
    
    print("Interactive RAG mode with Langfuse tracing")
    print("Type 'quit' to exit\n")
    
    session_id = f"interactive-{args.user_id}" if args.user_id else "interactive"
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        result = pipeline.query(question, user_id=args.user_id, session_id=session_id)
        
        print(f"\nAnswer:\n{result.answer}")
        print(f"\nSources:")
        for src in result.sources:
            print(f"  - [{src['book']}] {src['section']} (score: {src['score']})")
        print()
    
    pipeline.flush()
    print("Session ended. Traces sent to Langfuse.")


def main():
    parser = argparse.ArgumentParser(description="Lab 3: RAG with Langfuse")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # demo
    demo_parser = subparsers.add_parser("demo", help="Run demo queries")
    demo_parser.add_argument("-q", "--question", type=str, help="Single question")
    demo_parser.add_argument("--user-id", default="demo-user")
    demo_parser.add_argument("--session-id", default="demo-session")
    
    # create-dataset
    subparsers.add_parser("create-dataset", help="Create dataset in Langfuse")
    
    # experiment
    exp_parser = subparsers.add_parser("experiment", help="Run single experiment")
    exp_parser.add_argument("--name", default="experiment-default", help="Experiment name")
    exp_parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval")
    
    # compare
    subparsers.add_parser("compare", help="Run comparison experiments")
    
    # interactive
    int_parser = subparsers.add_parser("interactive", help="Interactive mode")
    int_parser.add_argument("--user-id", default="interactive-user")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        run_demo(args)
    elif args.command == "create-dataset":
        run_create_dataset(args)
    elif args.command == "experiment":
        run_experiment(args)
    elif args.command == "compare":
        run_comparison(args)
    elif args.command == "interactive":
        run_interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
