import yaml
import argparse
import requests
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.retrievers import EnsembleRetriever # Not available/working
from langchain_community.retrievers import BM25Retriever
import pickle
import os

class SimpleEnsembleRetriever:
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights or [1.0/len(retrievers)] * len(retrievers)

    def invoke(self, query):
        # Reciprocal Rank Fusion (RRF)
        all_docs = {}
        for i, retriever in enumerate(self.retrievers):
            try:
                docs = retriever.invoke(query)
            except Exception as e:
                print(f"Error in retriever {i}: {e}")
                continue
                
            for rank, doc in enumerate(docs):
                # Use page_content as unique key for deduplication
                if doc.page_content not in all_docs:
                    # Initialize score in metadata if not present (though we calculate it here)
                    if 'score' not in doc.metadata:
                        doc.metadata['score'] = 0.0
                    all_docs[doc.page_content] = doc
                
                # RRF score calculation: 1 / (k + rank)
                # k is a constant, typically 60
                k = 60
                score = 1.0 / (k + rank)
                all_docs[doc.page_content].metadata['score'] += score * self.weights[i]
        
        # Sort by accumulated score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x.metadata.get('score', 0), reverse=True)
        return sorted_docs

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

class RAGAgent:
    def __init__(self):
        self.config = self.load_config()
        self.retriever = self.load_retriever()

    def load_config(self):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def load_retriever(self):
        vector_type = self.config["vectorization"]["type"]
        vector_store_path = str(Path(__file__).parent.parent / self.config["vector_store"]["path"])
        
        # Helper to get dense retriever
        def get_dense_retriever():
            embedding_function = SentenceTransformerEmbeddings(model_name=self.config["embedding"]["model_name"])
            vector_store = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embedding_function,
                collection_name=self.config["vector_store"]["collection_name"]
            )
            return vector_store.as_retriever(search_kwargs={"k": 10})

        # Helper to get sparse retriever
        def get_sparse_retriever():
            bm25_path = os.path.join(vector_store_path, "bm25_retriever.pkl")
            if not os.path.exists(bm25_path):
                raise FileNotFoundError(f"BM25 retriever not found at {bm25_path}. Please rebuild index.")
            with open(bm25_path, "rb") as f:
                retriever = pickle.load(f)
                retriever.k = 10
                return retriever

        if vector_type == "dense":
            return get_dense_retriever()
        elif vector_type == "sparse":
            return get_sparse_retriever()
        elif vector_type == "hybrid":
            dense = get_dense_retriever()
            sparse = get_sparse_retriever()
            return SimpleEnsembleRetriever(
                retrievers=[dense, sparse],
                weights=[0.5, 0.5]
            )
        else:
            raise ValueError(f"Unknown vectorization type: {vector_type}")

    def retrieve(self, query, k=10):
        # Note: k is currently fixed in the retriever initialization for simplicity in this iteration
        # To make it dynamic, we'd need to adjust search_kwargs on the fly
        docs = self.retriever.invoke(query)
        return docs[:k] # Limit to k here just in case

    def call_llm(self, prompt):
        url = self.config["llm"]["base_url"]
        payload = {
            "model": self.config["llm"]["model_id"],
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error calling LLM: {e}"

    def format_citation(self, doc):
        """Format a citation string from document metadata."""
        source = doc.metadata.get('source', 'Unknown')
        citation_parts = [source]
        
        # Add page number if available
        if 'page' in doc.metadata:
            page_num = doc.metadata.get('page', 0) + 1  # 0-indexed to 1-indexed
            citation_parts.append(f"Page {page_num}")
        elif 'Header 1' in doc.metadata and 'Page' in doc.metadata['Header 1']:
            # Extract page number from header if present
            citation_parts.append(doc.metadata['Header 1'])
        
        # Add slide number if available (for PPTX)
        if 'slide' in doc.metadata:
            slide_num = doc.metadata.get('slide', 0)
            citation_parts.append(f"Slide {slide_num}")
        
        # Add section headers
        headers = []
        if 'Header 1' in doc.metadata and 'Page' not in doc.metadata.get('Header 1', ''):
            headers.append(doc.metadata['Header 1'])
        if 'Header 2' in doc.metadata:
            headers.append(doc.metadata['Header 2'])
        if 'Header 3' in doc.metadata:
            headers.append(doc.metadata['Header 3'])
        
        if headers:
            citation_parts.append(' > '.join(headers))
        
        return ' | '.join(citation_parts)
    
    def format_context_with_citations(self, docs):
        """Format context with proper citations for each chunk."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            citation = self.format_citation(doc)
            # Truncate long content for context (keep first 500 chars)
            content = doc.page_content
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"[{i}] Source: {citation}\nContent: {content}")
        return "\n\n".join(context_parts)

    def query(self, question):
        # 1. Retrieve relevant chunks
        docs = self.retrieve(question)
        context = self.format_context_with_citations(docs)

        # 2. Construct Prompt
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context. Provide a detailed and comprehensive answer. If the answer is not in the context, say "I don't know".

When referencing information, cite the source using the format [N] where N is the source number.

Context:
{context}

Question: {question}

Answer:"""

        # 3. Get Answer
        answer = self.call_llm(prompt)
        
        # 4. Format sources with citations
        sources = []
        citations = []
        for i, doc in enumerate(docs, 1):
            citation = self.format_citation(doc)
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
            "citations": citations
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    agent = RAGAgent()
    result = agent.query(args.query)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    
    print("\n" + "="*60)
    print("Sources and Citations:")
    print("="*60)
    for citation in result['citations']:
        print(f"\n[{citation['index']}] {citation['citation']}")
        print(f"    Snippet: {citation['snippet']}")
