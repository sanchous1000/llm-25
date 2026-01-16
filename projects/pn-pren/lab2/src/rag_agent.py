from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

import config


class RAGAgent:
    def __init__(self, collection_name=None):
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.ollama_base = f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}"
    
    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = config.TOP_K
        
        query_vector = self.embedder.encode([query])[0].tolist()
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        contexts = []
        for hit in results:
            contexts.append({
                'text': hit.payload['text'],
                'source': hit.payload['original_file'],
                'chunk_id': hit.payload['chunk_id'],
                'score': hit.score
            })
        
        return contexts
    
    def generate_answer(self, query, contexts):
        context_text = "\n\n---\n\n".join([
            f"[{ctx['source']}, chunk {ctx['chunk_id']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.
Always cite your sources by mentioning the document name and chunk number.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            response = ollama.chat(
                model=config.LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'base_url': self.ollama_base}
            )
            
            answer = response['message']['content']
            
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            answer = "Sorry, I couldn't generate an answer."
        
        return answer
    
    def query(self, question, top_k=None):
        print(f"\nQuery: {question}")
        print("Retrieving contexts...")
        
        contexts = self.retrieve(question, top_k)
        
        print(f"Found {len(contexts)} relevant chunks")
        
        print("Generating answer...")
        answer = self.generate_answer(question, contexts)
        
        result = {
            'question': question,
            'answer': answer,
            'contexts': contexts
        }
        
        return result
    
    def print_result(self, result):
        print("\n" + "="*60)
        print(f"Q: {result['question']}")
        print("-"*60)
        print(f"A: {result['answer']}")
        print("-"*60)
        print("Sources:")
        for ctx in result['contexts']:
            print(f"  - {ctx['source']} (chunk {ctx['chunk_id']}, score: {ctx['score']:.4f})")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help='Query to ask')
    parser.add_argument('--collection', default=config.COLLECTION_NAME)
    parser.add_argument('--top-k', type=int, default=config.TOP_K)
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    
    agent = RAGAgent(collection_name=args.collection)
    
    if args.interactive:
        print("RAG Agent (type 'exit' to quit)")
        while True:
            query = input("\n> ")
            if query.lower() in ['exit', 'quit']:
                break
            
            result = agent.query(query, top_k=args.top_k)
            agent.print_result(result)
    
    elif args.query:
        result = agent.query(args.query, top_k=args.top_k)
        agent.print_result(result)
    
    else:
        print("Use --query or --interactive")


if __name__ == '__main__':
    main()
