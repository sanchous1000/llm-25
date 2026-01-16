from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    def __init__(self, 
                 qdrant_url="http://localhost:6333", 
                 collection_name="verl_rag",
                 llm_base_url="http://localhost:8080/v1",
                 emb_model="intfloat/e5-large-v2",
                 top_k=5):
        
        self.collection_name = collection_name
        self.top_k = top_k
        
        self.embeddings = HuggingFaceEmbeddings(model_name=emb_model)
        
        self.client = QdrantClient(url=qdrant_url)

        self.llm = ChatOpenAI(
            base_url=llm_base_url,
            api_key="sk-no-key-required",
            model="llama-3.2-3b-instruct",
            temperature=0.1,
            max_tokens=256
        )

        template = """You are an expert AI assistant.
Use the retrieved context to answer. If unsure, say "I don't know".
ALWAYS cite the source filename.

Context:
{context}

Question:
{question}

Answer:"""
        self.prompt = ChatPromptTemplate.from_template(template)

    def format_search_results(self, points):
        formatted = []
        for point in points:
            payload = point.payload or {}
            source = payload.get('source', 'unknown_source')
            content = payload.get('page_content') or payload.get('text', '')
            
            content = content.replace("\n", " ")
            formatted.append(f"[Source: {source}]\n{content}\n")
        return "\n".join(formatted)

    def answer_without_rag(self, item):
        question = item.input if hasattr(item, 'input') else item
        ground_truth = item.expected_output if hasattr(item, 'expected_output') else None
        chain = (
            ChatPromptTemplate.from_template("Answer this question: {question}")
            | self.llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        
        return {
            "answer": answer,
            "retrieved_contexts": [],
            "retrieved_sources": [],
            "ground_truth": ground_truth,
        }

    def answer_question(self, item):
        question = item.input if hasattr(item, 'input') else item
        query_vector = self.embeddings.embed_query(question)
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=self.top_k
        )

        context_str = self.format_search_results(search_result.points)
        
        chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        return {
            "answer": answer,
            "retrieved_contexts": [
                (p.payload.get('page_content') or p.payload.get('text', '')) for p in search_result.points
            ],
            "retrieved_sources": [
                p.payload.get('source', 'unknown') for p in search_result.points
            ],
            "ground_truth": item.expected_output if hasattr(item, 'expected_output') else None,
        }