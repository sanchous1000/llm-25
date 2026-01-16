import argparse
from src.config import IndexConfig
from src.indexer import Indexer
from src.evaluation import evaluate_retrieval
from src.rag_engine import RAGService

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Manager")
    parser.add_argument("--mode", choices=["index", "eval", "chat"], required=True)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--query", type=str, help="Query for chat mode")
    
    args = parser.parse_args()
    
    config = IndexConfig(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    
    if args.mode == "index":
        indexer = Indexer(config)
        indexer.rebuild_index("data/processed")
        
    elif args.mode == "eval":
        evaluate_retrieval(config)
        
    elif args.mode == "chat":
        if not args.query:
            print("Please provide --query")
            return
        rag = RAGService(config)
        response = rag.answer_question(args.query)
        print(f"\nFinal Answer:\n{response}")

if __name__ == "__main__":
    main()
