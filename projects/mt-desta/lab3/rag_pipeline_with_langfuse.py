"""
Langfuse Integration Wrapper for Lab2 RAG Pipeline
Wraps the existing lab2 RAGAgent with Langfuse logging
Enhanced with user session tracking and comprehensive timestamps
"""
import os
import argparse
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import requests

# Langfuse imports
from langfuse import Langfuse
from dotenv import load_dotenv

# Import the existing RAGAgent from lab2
import sys
sys.path.append(str(Path(__file__).parent.parent / "lab2" / "scripts"))
from rag_pipeline import RAGAgent

load_dotenv()


class LangfuseRAGAgent:
    """
    Wrapper around lab2's RAGAgent that adds Langfuse logging.
    Uses the existing RAGAgent from lab2 without modifying it.
    Enhanced with user session tracking and comprehensive timestamps.
    """
    
    def __init__(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        # Use the existing RAGAgent from lab2
        self.rag_agent = RAGAgent()
        
        # Initialize Langfuse client with error handling
        try:
            self.langfuse = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "http://localhost:3001")
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Langfuse: {e}")
            print("Continuing without Langfuse logging...")
            self.langfuse = None
        
        # User and session tracking
        self.user_id = user_id or os.getenv("USER_ID", "default-user")
        self.session_id = session_id or os.getenv("SESSION_ID") or str(uuid.uuid4())
        self.config = self.rag_agent.config
        
        # Track session start time
        self.session_start_time = datetime.now(timezone.utc)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Main RAG query method with comprehensive Langfuse tracing.
        Creates a trace that links all steps: retrieval, prompt construction, and LLM call.
        Enhanced with user session tracking and timestamps.
        """
        # If Langfuse is not available, run without logging
        if self.langfuse is None:
            return self._query_without_logging(question)
        
        # Track query start time
        query_start_time = datetime.now(timezone.utc)
        
        # Create main trace using SDK v3 API
        try:
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="rag_query",
                metadata={
                    "user_id": self.user_id,  # User ID in metadata for filtering
                    "session_id": self.session_id,  # Session ID for grouping
                    "retrieval_type": self.config["vectorization"]["type"],
                    "model": self.config["llm"]["model_id"],
                    "session_start_time": self.session_start_time.isoformat(),
                    "query_start_time": query_start_time.isoformat(),
                    "environment": os.getenv("ENVIRONMENT", "development")
                }
            ) as trace:
                # Step 1: Retrieve relevant chunks (with logging)
                retrieval_start_time = datetime.now(timezone.utc)
                with self.langfuse.start_as_current_observation(
                    as_type="generation",
                    name="document_retrieval"
                ) as retrieval_gen:
                    docs = self.rag_agent.retrieve(question)
                    context = self.rag_agent.format_context_with_citations(docs)
                    retrieval_end_time = datetime.now(timezone.utc)
                    retrieval_duration = (retrieval_end_time - retrieval_start_time).total_seconds()
                    
                    # Log retrieval details
                    retrieval_metadata = []
                    for i, doc in enumerate(docs):
                        retrieval_metadata.append({
                            "rank": i + 1,
                            "source": doc.metadata.get("source", "Unknown"),
                            "score": doc.metadata.get("score", 0.0),
                            "chunk_length": len(doc.page_content),
                            "preview": doc.page_content[:200]
                        })
                    
                    retrieval_gen.update(
                        input=question,
                        output={
                            "num_documents": len(docs),
                            "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
                        },
                        metadata={
                            "retrieval_type": self.config["vectorization"]["type"],
                            "k": len(docs),
                            "documents": retrieval_metadata,
                            "start_time": retrieval_start_time.isoformat(),
                            "end_time": retrieval_end_time.isoformat(),
                            "duration_seconds": retrieval_duration
                        }
                    )
                
                # Step 2: Construct Prompt (same as lab2)
                prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context. Provide a detailed and comprehensive answer. If the answer is not in the context, say "I don't know".

When referencing information, cite the source using the format [N] where N is the source number.

Context:
{context}

Question: {question}

Answer:"""
                
                # Step 3: Get Answer from LLM (with logging)
                llm_start_time = datetime.now(timezone.utc)
                with self.langfuse.start_as_current_observation(
                    as_type="generation",
                    name="llm_generation",
                    model=self.config["llm"]["model_id"]
                ) as llm_gen:
                    # Call LLM using the existing method
                    answer = self.rag_agent.call_llm(prompt)
                    
                    llm_end_time = datetime.now(timezone.utc)
                    llm_duration = (llm_end_time - llm_start_time).total_seconds()
                    
                    # Log LLM call details with comprehensive timestamps
                    llm_gen.update(
                        input=prompt,
                        output=answer,
                        model_parameters={
                            "base_url": self.config["llm"]["base_url"],
                            "stream": False
                        },
                        metadata={
                            "start_time": llm_start_time.isoformat(),
                            "end_time": llm_end_time.isoformat(),
                            "duration_seconds": llm_duration,
                            "response_length": len(answer),
                            "prompt_length": len(prompt),
                            "context_length": len(context),
                            "tokens_estimated": len(prompt.split()) + len(answer.split())  # Rough estimate
                        }
                    )
                
                # Step 4: Format sources with citations (same as lab2)
                sources = []
                citations = []
                for i, doc in enumerate(docs, 1):
                    citation = self.rag_agent.format_citation(doc)
                    sources.append(citation)
                    citations.append({
                        "index": i,
                        "citation": citation,
                        "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    })
                
                # Calculate total query duration
                query_end_time = datetime.now(timezone.utc)
                total_duration = (query_end_time - query_start_time).total_seconds()
                
                # End trace with final result - set both input and output explicitly
                trace.update(
                    input={"question": question},
                    output={
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                        "num_sources": len(sources)
                    },
                    metadata={
                        "query_start_time": query_start_time.isoformat(),
                        "query_end_time": query_end_time.isoformat(),
                        "total_duration_seconds": total_duration,
                        "retrieval_duration_seconds": retrieval_duration,
                        "llm_duration_seconds": llm_duration,
                        "num_retrieved_docs": len(docs),
                        "session_id": self.session_id,
                        "user_id": self.user_id,
                        "timestamp": query_end_time.isoformat()
                    }
                )
                
                trace_id = trace.id
            
            # Flush to ensure traces are sent to Langfuse immediately
            try:
                self.langfuse.flush()
                print(f"Trace sent to Langfuse. Trace ID: {trace_id}")
            except Exception as flush_error:
                print(f"Warning: Failed to flush trace: {flush_error}")
        except Exception as e:
            print(f"Warning: Langfuse logging failed: {e}")
            print("Continuing without Langfuse logging...")
            return self._query_without_logging(question)
        
        return {
            "question": question,
            "answer": answer,
            "context": [doc.page_content for doc in docs],
            "sources": sources,
            "citations": citations,
            "trace_id": trace_id
        }
    
    def retrieve(self, query: str, k: int = 10):
        """Wrapper for retrieval with Langfuse logging."""
        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="retrieve_documents",
            metadata={"query": query, "k": k, "user_id": self.user_id}
        ) as trace:
            docs = self.rag_agent.retrieve(query, k)
            
            trace.update(
                input=query,
                output={"num_documents": len(docs)},
                metadata={
                    "retrieval_type": self.config["vectorization"]["type"],
                    "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
                }
            )
        
        return docs
    
    def _query_without_logging(self, question: str) -> Dict[str, Any]:
        """Fallback query method without Langfuse logging."""
        # Step 1: Retrieve relevant chunks
        docs = self.rag_agent.retrieve(question)
        context = self.rag_agent.format_context_with_citations(docs)
        
        # Step 2: Construct Prompt
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context. Provide a detailed and comprehensive answer. If the answer is not in the context, say "I don't know".

When referencing information, cite the source using the format [N] where N is the source number.

Context:
{context}

Question: {question}

Answer:"""
        
        # Step 3: Get Answer from LLM
        answer = self.rag_agent.call_llm(prompt)
        
        # Step 4: Format sources with citations
        sources = []
        citations = []
        for i, doc in enumerate(docs, 1):
            citation = self.rag_agent.format_citation(doc)
            sources.append(citation)
            citations.append({
                "index": i,
                "citation": citation,
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "question": question,
            "answer": answer,
            "context": [doc.page_content for doc in docs],
            "sources": sources,
            "citations": citations,
            "trace_id": None
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline with Langfuse Logging")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--user-id", type=str, help="User ID for session tracking")
    parser.add_argument("--session-id", type=str, help="Session ID for grouping queries")
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = LangfuseRAGAgent(user_id=args.user_id, session_id=args.session_id)
    result = agent.query(args.query)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nTrace ID: {result['trace_id']}")
    print(f"Session ID: {agent.session_id}")
    print(f"User ID: {agent.user_id}")
    
    print("\n" + "="*60)
    print("Sources and Citations:")
    print("="*60)
    for citation in result['citations']:
        print(f"\n[{citation['index']}] {citation['citation']}")
        print(f"    Snippet: {citation['snippet']}")

