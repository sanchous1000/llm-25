"""
RAG Agent with Langfuse integration for logging and tracing.
Logs all retrieval and generation steps with metadata.
"""
import time
import uuid
from typing import List, Dict, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

import config


class LangfuseRAGAgent:
    """RAG Agent with full Langfuse tracing for all operations."""
    
    def __init__(self, collection_name: str = None, user_id: str = None):
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=config.QDRANT_HOST, 
            port=config.QDRANT_PORT
        )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Ollama base URL
        self.ollama_base = f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}"
        
        # Initialize Langfuse client
        self.langfuse = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST
        )
        
        print(f"RAG Agent initialized for user: {self.user_id}")
        print(f"Langfuse host: {config.LANGFUSE_HOST}")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = None,
        trace_id: str = None,
        parent_observation_id: str = None
    ) -> tuple[List[Dict], str]:
        """
        Retrieve relevant documents from vector store.
        Returns contexts and span_id for tracing.
        """
        top_k = top_k or config.TOP_K
        start_time = time.time()
        
        # Create span for retrieval
        span = self.langfuse.span(
            name="retrieval",
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            input={"query": query, "top_k": top_k},
            metadata={
                "collection": self.collection_name,
                "embedding_model": config.EMBEDDING_MODEL
            }
        )
        
        try:
            # Encode query
            embed_start = time.time()
            query_vector = self.embedder.encode([query])[0].tolist()
            embed_duration = time.time() - embed_start
            
            # Search in Qdrant
            search_start = time.time()
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            ).points
            search_duration = time.time() - search_start
            
            # Format contexts
            contexts = []
            for hit in results:
                contexts.append({
                    'text': hit.payload.get('text', ''),
                    'source': hit.payload.get('original_file', 'unknown'),
                    'chunk_id': hit.payload.get('chunk_id', hit.id),
                    'score': hit.score
                })
            
            total_duration = time.time() - start_time
            
            # Update span with results
            span.end(
                output={
                    "num_results": len(contexts),
                    "top_scores": [ctx['score'] for ctx in contexts[:3]],
                    "sources": list(set(ctx['source'] for ctx in contexts))
                },
                metadata={
                    "embed_duration_ms": round(embed_duration * 1000, 2),
                    "search_duration_ms": round(search_duration * 1000, 2),
                    "total_duration_ms": round(total_duration * 1000, 2)
                }
            )
            
            return contexts, span.id
            
        except Exception as e:
            span.end(
                level="ERROR",
                status_message=str(e)
            )
            raise
    
    def generate_answer(
        self, 
        query: str, 
        contexts: List[Dict],
        trace_id: str = None,
        parent_observation_id: str = None
    ) -> tuple[str, Dict]:
        """
        Generate answer using LLM with retrieved contexts.
        Returns answer and generation metadata.
        """
        start_time = time.time()
        
        # Build context text
        context_text = "\n\n---\n\n".join([
            f"[{ctx['source']}, chunk {ctx['chunk_id']}]\n{ctx['text']}"
            for ctx in contexts
        ])
        
        # Build prompt
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.
Always cite your sources by mentioning the document name and chunk number.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Create generation span
        generation = self.langfuse.generation(
            name="llm_generation",
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            model=config.LLM_MODEL,
            input={"prompt": prompt, "query": query},
            metadata={
                "num_contexts": len(contexts),
                "context_chars": len(context_text)
            }
        )
        
        try:
            # Call Ollama
            response = ollama.chat(
                model=config.LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'base_url': self.ollama_base}
            )
            
            answer = response['message']['content']
            
            # Extract token usage if available
            usage = {
                "prompt_tokens": response.get('prompt_eval_count', 0),
                "completion_tokens": response.get('eval_count', 0),
                "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
            }
            
            duration = time.time() - start_time
            
            # End generation with results
            generation.end(
                output=answer,
                usage=usage,
                metadata={
                    "duration_ms": round(duration * 1000, 2),
                    "answer_length": len(answer)
                }
            )
            
            return answer, {
                "usage": usage,
                "duration_ms": round(duration * 1000, 2),
                "model": config.LLM_MODEL
            }
            
        except Exception as e:
            generation.end(
                level="ERROR",
                status_message=str(e)
            )
            return f"Error generating answer: {str(e)}", {"error": str(e)}
    
    def query(
        self, 
        question: str, 
        top_k: int = None,
        session_id: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Full RAG query with complete Langfuse tracing.
        """
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create trace for the entire query
        trace = self.langfuse.trace(
            name="rag_query",
            user_id=self.user_id,
            session_id=session_id,
            input={"question": question, "top_k": top_k or config.TOP_K},
            metadata={
                "collection": self.collection_name,
                "llm_model": config.LLM_MODEL,
                "embedding_model": config.EMBEDDING_MODEL,
                **(metadata or {})
            }
        )
        
        try:
            print(f"\nQuery: {question}")
            print("Retrieving contexts...")
            
            # Retrieval step
            contexts, retrieval_span_id = self.retrieve(
                query=question,
                top_k=top_k,
                trace_id=trace.id
            )
            
            print(f"Found {len(contexts)} relevant chunks")
            print("Generating answer...")
            
            # Generation step
            answer, gen_metadata = self.generate_answer(
                query=question,
                contexts=contexts,
                trace_id=trace.id
            )
            
            # Build result
            result = {
                'question': question,
                'answer': answer,
                'contexts': contexts,
                'trace_id': trace.id,
                'session_id': session_id,
                'metadata': {
                    'retrieval': {
                        'num_contexts': len(contexts),
                        'top_k': top_k or config.TOP_K
                    },
                    'generation': gen_metadata
                }
            }
            
            # Update trace with final output
            trace.update(
                output={
                    "answer": answer,
                    "num_contexts": len(contexts),
                    "sources": list(set(ctx['source'] for ctx in contexts))
                }
            )
            
            return result
            
        except Exception as e:
            trace.update(
                output={"error": str(e)},
                metadata={"status": "error"}
            )
            raise
        finally:
            # Ensure all data is sent to Langfuse
            self.langfuse.flush()
    
    def score_trace(self, trace_id: str, name: str, value: float, comment: str = None):
        """Add a score to a trace for evaluation."""
        self.langfuse.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment
        )
        self.langfuse.flush()
    
    def print_result(self, result: Dict):
        """Pretty print query result."""
        print("\n" + "=" * 60)
        print(f"Q: {result['question']}")
        print("-" * 60)
        print(f"A: {result['answer']}")
        print("-" * 60)
        print("Sources:")
        for ctx in result['contexts']:
            print(f"  - {ctx['source']} (chunk {ctx['chunk_id']}, score: {ctx['score']:.4f})")
        print(f"\nTrace ID: {result['trace_id']}")
        print(f"Session ID: {result['session_id']}")
        print("=" * 60)
    
    def close(self):
        """Flush and close Langfuse client."""
        self.langfuse.flush()
        self.langfuse.shutdown()


def main():
    """Interactive RAG agent demo with Langfuse logging."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Agent with Langfuse logging')
    parser.add_argument('--query', type=str, help='Query to ask')
    parser.add_argument('--collection', default=config.COLLECTION_NAME)
    parser.add_argument('--top-k', type=int, default=config.TOP_K)
    parser.add_argument('--user-id', type=str, help='User ID for session tracking')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    args = parser.parse_args()
    
    agent = LangfuseRAGAgent(
        collection_name=args.collection,
        user_id=args.user_id
    )
    
    try:
        if args.interactive:
            print("RAG Agent with Langfuse Logging (type 'exit' to quit)")
            session_id = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            while True:
                query = input("\n> ")
                if query.lower() in ['exit', 'quit']:
                    break
                
                result = agent.query(query, top_k=args.top_k, session_id=session_id)
                agent.print_result(result)
        
        elif args.query:
            result = agent.query(args.query, top_k=args.top_k)
            agent.print_result(result)
        
        else:
            print("Use --query or --interactive")
    
    finally:
        agent.close()


if __name__ == '__main__':
    main()
