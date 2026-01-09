"""
RAG-пайплайн для ответов на вопросы с использованием векторного поиска и LLM.
Выполняет задание 6: реализация движка общения (RAG-пайплайн).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from openai import OpenAI

from evaluate_retrieval import RetrievalEvaluator
from utils import get_device


class RAGPipeline:
    """Класс для реализации RAG-пайплайна."""
    
    def __init__(self,
                 faiss_index_dir: str,
                 chunks_dir: str,
                 dense_model_name: str,
                 llm_client: OpenAI,
                 llm_model: str = "qwen2.5:7b",
                 search_type: str = "hybrid",
                 top_k: int = 5,
                 device: Optional[str] = None):
        """
        Инициализация RAG-пайплайна.
        
        Args:
            faiss_index_dir: Директория с индексом Faiss
            chunks_dir: Директория с чанками
            dense_model_name: Название модели для dense эмбеддингов
            llm_client: Клиент OpenAI для вызова LLM
            llm_model: Название модели LLM
            search_type: Тип поиска (dense, sparse, hybrid)
            top_k: Количество релевантных чанков для контекста
            device: Устройство для вычислений
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.search_type = search_type
        self.top_k = top_k
        
        # Инициализируем оценщик для поиска
        self.evaluator = RetrievalEvaluator(
            faiss_index_dir,
            chunks_dir,
            dense_model_name,
            device=device
        )
        
        # Системный промпт
        self.system_prompt = """Ты - полезный ассистент, который отвечает на вопросы на основе предоставленной документации.
Используй только информацию из предоставленных контекстов для ответа.
Если в контекстах нет информации для ответа на вопрос, честно скажи об этом.
Всегда указывай источники информации в своем ответе, ссылаясь на конкретные документы.
Отвечай на русском языке, четко и структурированно."""
    
    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Извлекает релевантные чанки для запроса.
        
        Args:
            query: Вопрос пользователя
            
        Returns:
            Список релевантных чанков с метаданными
        """
        if self.search_type == "dense":
            results = self.evaluator.search_dense(query, k=self.top_k)
        elif self.search_type == "sparse":
            results = self.evaluator.search_sparse(query, k=self.top_k)
        elif self.search_type == "hybrid":
            results = self.evaluator.search_hybrid(query, k=self.top_k)
        else:
            raise ValueError(f"Неизвестный тип поиска: {self.search_type}")
        
        return results
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Форматирует чанки в текстовый контекст для промпта.
        
        Args:
            chunks: Список чанков с метаданными
            
        Returns:
            Отформатированный контекст
        """
        context_parts = []
        
        for i, result in enumerate(chunks, start=1):
            chunk = result.get("chunk", {})
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            # Извлекаем метаданные
            source_file = metadata.get("source_file", "Неизвестный источник")
            repository = metadata.get("repository", "Неизвестный репозиторий")
            repository_url = metadata.get("repository_url", "")
            header = metadata.get("Header 1", "")
            
            # Формируем заголовок контекста
            context_header = f"[Контекст {i}]"
            if header:
                context_header += f" Раздел: {header}"
            if repository:
                context_header += f" | Репозиторий: {repository}"
            if repository_url:
                context_header += f" ({repository_url})"
            
            context_parts.append(f"{context_header}\n{text}\n")
        
        return "\n---\n\n".join(context_parts)
    
    def build_prompt(self, question: str, context: str) -> List[Dict[str, str]]:
        """
        Собирает промпт для LLM.
        
        Args:
            question: Вопрос пользователя
            context: Контекст из релевантных чанков
            
        Returns:
            Список сообщений для OpenAI API
        """
        user_prompt = f"""Вопрос: {question}

Контекст из документации:
{context}

Ответь на вопрос, используя информацию из предоставленного контекста. Укажи источники информации."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def generate_answer(self, question: str, temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Генерирует ответ на вопрос с использованием RAG.
        
        Args:
            question: Вопрос пользователя
            temperature: Температура для генерации
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Словарь с ответом, контекстом и метаданными
        """
        print(f"🔍 Поиск релевантных документов...")
        retrieved_chunks = self.retrieve_context(question)
        
        if not retrieved_chunks:
            return {
                "answer": "Извините, не удалось найти релевантную информацию для вашего вопроса.",
                "context": [],
                "sources": []
            }
        
        print(f"✅ Найдено {len(retrieved_chunks)} релевантных фрагментов")
        
        context_text = self.format_context(retrieved_chunks)
        
        messages = self.build_prompt(question, context_text)
        
        print(f"🤖 Генерация ответа с помощью Ollama ({self.llm_model})...")
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"repeat_penalty": 1.1}
            )
            
            answer = response.choices[0].message.content
            
        except Exception as e:
            return {
                "answer": f"Ошибка при генерации ответа: {str(e)}",
                "context": retrieved_chunks,
                "sources": self._extract_sources(retrieved_chunks)
            }
        
        formatted_answer = self._format_answer_with_citations(answer, retrieved_chunks)
        
        return {
            "answer": formatted_answer,
            "raw_answer": answer,
            "context": retrieved_chunks,
            "sources": self._extract_sources(retrieved_chunks)
        }
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Извлекает источники из чанков."""
        sources = []
        for chunk_result in chunks:
            chunk = chunk_result.get("chunk", {})
            metadata = chunk.get("metadata", {})
            
            source = {
                "repository": metadata.get("repository", "Неизвестно"),
                "repository_url": metadata.get("repository_url", ""),
                "source_file": metadata.get("source_file", ""),
                "header": metadata.get("Header 1", ""),
                "snippet": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "")
            }
            sources.append(source)
        
        return sources
    
    def _format_answer_with_citations(self, answer: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Форматирует ответ с цитатами источников.
        
        Args:
            answer: Ответ от LLM
            chunks: Релевантные чанки
            
        Returns:
            Отформатированный ответ с цитатами
        """
        sources_section = "\n\n---\n\n**Источники:**\n\n"
        
        for i, chunk_result in enumerate(chunks, start=1):
            chunk = chunk_result.get("chunk", {})
            metadata = chunk.get("metadata", {})
            
            repository = metadata.get("repository", "Неизвестный репозиторий")
            repository_url = metadata.get("repository_url", "")
            header = metadata.get("Header 1", "")
            
            source_text = f"{i}. "
            if repository_url:
                source_text += f"[{repository}]({repository_url})"
            else:
                source_text += repository
            
            if header:
                source_text += f" — {header}"
            
            sources_section += source_text + "\n"
        
        return answer + sources_section


def create_llm_client() -> OpenAI:
    """
    Создает клиент для Ollama API.
    Всегда использует Ollama (как в lab1).
    
    Returns:
        Клиент OpenAI для Ollama
    """
    # Всегда используем Ollama (как в lab1)
    api_base = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    
    client = OpenAI(
        base_url=api_base,
        api_key=api_key
    )
    
    return client


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG-пайплайн для ответов на вопросы')
    parser.add_argument('--faiss-index-dir', type=str, default='faiss_index',
                       help='Директория с индексом Faiss')
    parser.add_argument('--chunks-dir', type=str, default='chunks',
                       help='Директория с чанками')
    parser.add_argument('--dense-model', type=str, default='intfloat/multilingual-e5-large',
                       help='Модель для dense эмбеддингов')
    parser.add_argument('--llm-model', type=str, default='qwen2.5:7b',
                       help='Название модели LLM из Ollama (по умолчанию qwen2.5:7b, как в lab1)')
    parser.add_argument('--search-type', type=str, choices=['dense', 'sparse', 'hybrid'],
                       default='hybrid', help='Тип поиска')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Количество релевантных чанков для контекста')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений (mps/cuda/cpu)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Температура для генерации')
    parser.add_argument('--max-tokens', type=int, default=1000,
                       help='Максимальное количество токенов в ответе')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RAG-пайплайн для ответов на вопросы")
    print("=" * 60)
    
    # Создаем клиент LLM для Ollama (как в lab1)
    print(f"\n🔧 Инициализация Ollama клиента...")
    llm_client = create_llm_client()
    api_base_used = os.getenv('LLM_API_BASE', 'http://localhost:11434/v1')
    print(f"✅ Ollama клиент создан (API: {api_base_used})")
    
    # Инициализируем RAG-пайплайн
    print(f"\n🔧 Инициализация RAG-пайплайна...")
    device = get_device(args.device) if args.device else None
    rag = RAGPipeline(
        faiss_index_dir=args.faiss_index_dir,
        chunks_dir=args.chunks_dir,
        dense_model_name=args.dense_model,
        llm_client=llm_client,
        llm_model=args.llm_model,
        search_type=args.search_type,
        top_k=args.top_k,
        device=device
    )
    print("✅ RAG-пайплайн инициализирован")
    
    # Интерактивный режим
    print(f"\n{'='*60}")
    print("Интерактивный режим")
    print("Введите 'exit' или 'quit' для выхода")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n❓ Ваш вопрос: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'выход']:
                print("👋 До свидания!")
                break
            
            print()
            result = rag.generate_answer(
                question,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            print(f"\n{'='*60}")
            print("Ответ:")
            print("=" * 60)
            print(result["answer"])
            print()
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")