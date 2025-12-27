import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import click
import time
import uuid
sys.path.insert(0, str(Path(__file__).parent.parent))
from source.utils import load_config
from source.embeddings import DenseEmbedder
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv
from langfuse import Langfuse, get_client
load_dotenv()


class RAGPipeline:
    
    def __init__(self, config_path = './data/config.yaml'):
        self.config = load_config(config_path)
        qdrant_config = self.config.get('qdrant', {})
        self.host = qdrant_config.get('host', 'localhost')
        self.port = qdrant_config.get('port', 6333)
        self.collection_name = qdrant_config.get('collection_name', 'documents')
        self.ef_search = qdrant_config.get('hnsw', {}).get('ef_search', 100)
        
        # Параметры RAG
        rag_config = self.config.get('rag', {})
        self.top_k = rag_config.get('top_k', 5)
        self.llm_model = rag_config.get('llm_model', 'llama3.1:8b')
        self.temperature = rag_config.get('temperature', 0.7)
        self.max_tokens = rag_config.get('max_tokens', 512)
        
        # Ollama
        ollama_config = rag_config.get('ollama', {})
        self.ollama_base_url = ollama_config.get('base_url', 'http://localhost:11434/v1')
        self.ollama_api_key = ollama_config.get('api_key', 'pass')
        
        # Подключаемся к QDrant
        self.client = QdrantClient(host=self.host, port=self.port)
        # Инициализируем эмбеддер
        self.embedder = DenseEmbedder(self.config['embeddings']['dense'])
        
        # Инициализируем LLM клиент (Ollama через OpenAI API)
        self.llm_client = OpenAI(
            base_url=self.ollama_base_url,
            api_key=self.ollama_api_key
        )
        
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_host = os.getenv("LANGFUSE_BASE_URL")
        
        self.langfuse = Langfuse(
            secret_key=langfuse_secret_key,
            public_key=langfuse_public_key,
            base_url=langfuse_host
        )

    
    def search_relevant_chunks(self, query, top_k = None, trace = None):
        if top_k is None:
            top_k = self.top_k
        
        if trace is not None:
            span_context = trace.start_as_current_observation(
                as_type="span",
                name="search_relevant_chunks",
                input={"query": query, "top_k": top_k},
                metadata={"collection_name": self.collection_name}
            )
        else:
            from contextlib import nullcontext
            span_context = nullcontext()
        
        with span_context as span:
            # Создаем эмбеддинг для запроса
            model_name = self.config['embeddings']['dense']['model'].lower()
            
            if 'bge' in model_name:
                query_text = f"Represent this sentence for searching relevant passages: {query}"
            elif 'e5' in model_name:
                query_text = f"query: {query}"
            else:
                query_text = query
            
            embeddings_result = self.embedder.embed_texts([query_text])
            query_vector = embeddings_result['dense'][0].tolist()
            
            # Выполняем поиск в QDrant
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True
            ).points
            
            # Формируем результаты
            chunks = []
            for i, result in enumerate(results):
                payload = result.payload
                chunks.append({
                    'rank': i + 1,
                    'score': result.score,
                    'document': payload.get('document', 'unknown'),
                    'chunk_id': payload.get('chunk_id', 'unknown'),
                    'text': payload.get('text', ''),
                    'metadata': payload.get('metadata', {})
                })
            
            # Обновляем span с выходными данными (включая полные метаданные)
            if trace is not None and span is not None:
                span.update(
                    output={
                        "num_chunks": len(chunks),
                        "chunks": [
                            {
                                "rank": c["rank"],
                                "score": c["score"],
                                "document": c["document"],
                                "chunk_id": c["chunk_id"],
                                "text": c["text"],
                                "metadata": c["metadata"]
                            }
                            for c in chunks
                        ]
                    }
                )
            
            return chunks


    def build_prompt(self, query, chunks, trace = None):
        if trace is not None:
            span_context = trace.start_as_current_observation(
                as_type="span",
                name="build_prompt",
                input={"query": query, "num_chunks": len(chunks)},
                metadata={
                    "chunk_documents": [c["document"] for c in chunks],
                    "chunk_scores": [c["score"] for c in chunks]
                }
            )
        else:
            from contextlib import nullcontext
            span_context = nullcontext()
        
        with span_context as span:
            # Системный промпт с инструкциями
            system_instruction = """Ты - полезный ассистент для ответов на вопросы на основе предоставленных документов.

Твоя задача:
1. Внимательно изучи предоставленные фрагменты документов (контекст).
2. Ответь на вопрос пользователя, опираясь ТОЛЬКО на информацию из контекста.
3. Если информации недостаточно для ответа, честно скажи об этом.
4. В конце ответа ОБЯЗАТЕЛЬНО укажи источники: из каких документов взята информация.
5. Формат ответа должен быть структурированным и понятным."""

            # Формируем контексты из чанков
            contexts = []
            for chunk in chunks:
                doc_name = chunk['document']
                text = chunk['text']
                metadata = chunk['metadata']
                # Дополнительная информация из метаданных
                extra_info = []
                if 'page' in metadata:
                    extra_info.append(f"Страница: {metadata['page']}")
                if 'slide' in metadata:
                    extra_info.append(f"Слайд: {metadata['slide']}")
                if 'section' in metadata:
                    extra_info.append(f"Раздел: {metadata['section']}")
                
                extra_str = ", ".join(extra_info) if extra_info else ""
                context_header = f"[Документ: {doc_name}"
                if extra_str:
                    context_header += f", {extra_str}"
                context_header += "]"
                contexts.append(f"{context_header}\n{text}")
            
            context_block = "\n\n---\n\n".join(contexts)
            # Собираем финальный промпт
            prompt = f"""{system_instruction}

=== КОНТЕКСТ ИЗ ДОКУМЕНТОВ ===

{context_block}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===

{query}

=== ОТВЕТ ===

"""
            
            # Обновляем span с информацией о промпте
            if trace is not None and span is not None:
                span.update(
                    output={
                        "prompt_length": len(prompt),
                        "context_length": len(context_block),
                        "num_contexts": len(contexts)
                    },
                    metadata={
                        "system_instruction_length": len(system_instruction)
                    }
                )
            
            return prompt

    
    def generate_answer(self, prompt, trace = None):  
        # Создаем generation для LLM вызова с контекстным менеджером (если trace предоставлен)
        if trace is not None:
            generation_context = trace.start_as_current_observation(
                as_type="generation",
                name="llm_generation",
                model=self.llm_model,
                model_parameters={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                input=prompt
            )
        else:
            from contextlib import nullcontext
            generation_context = nullcontext()
        
        with generation_context as generation:
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                usage = response.usage
                answer_text = response.choices[0].message.content.strip()
                
                # Обновляем generation с выходными данными, usage и метаданными
                if trace is not None and generation is not None:
                    generation.update(
                        output=answer_text,
                        usage={
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens
                        },
                        metadata={
                            "model": self.llm_model
                        }
                    )
                
                return {
                    'success': True,
                    'answer': answer_text,
                    'model': self.llm_model,
                    'tokens': {
                        'prompt': usage.prompt_tokens,
                        'completion': usage.completion_tokens,
                        'total': usage.total_tokens
                    },
                }
                
            except Exception as e:
                # Обновляем generation с информацией об ошибке
                if trace is not None and generation is not None:
                    generation.update(
                        output=None,
                        level="ERROR",
                        status_message=str(e)
                    )
                
                return {
                    'success': False,
                    'error': str(e),
                    'answer': None
                }


    
    def format_citations(self, chunks):
        citations = []
        
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk['document']
            metadata = chunk['metadata']
            text_snippet = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            citation = f"{i}. **{doc_name}**"
            
            # Дополнительная информация
            extras = []
            if 'page' in metadata:
                extras.append(f"страница {metadata['page']}")
            if 'slide' in metadata:
                extras.append(f"слайд {metadata['slide']}")
            if 'section' in metadata:
                extras.append(f"раздел '{metadata['section']}'")
            
            if extras:
                citation += f" ({', '.join(extras)})"
            
            citation += f"\n   Фрагмент: \"{text_snippet}\""
            citation += f"\n   Релевантность: {chunk['score']:.3f}"
            citations.append(citation)
        
        return "\n\n".join(citations)
    
    def answer_question(self, query, top_k = None, session_id = None):
        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="rag_query",
            input={"query": query, "top_k": top_k},
            metadata={
                "model": self.llm_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        ) as trace:
            # Привязываем trace к сессии
            if session_id:
                trace.update_trace(session_id=session_id)
            
            # 1. Поиск релевантных чанков
            chunks = self.search_relevant_chunks(query, top_k, trace)
            
            if not chunks:
                result = {
                    'query': query,
                    'answer': "К сожалению, я не нашел релевантных документов для ответа на ваш вопрос.",
                    'sources': [],
                    'success': False
                }
                trace.update(output=result, level="WARNING")
                return result
            
            # 2. Сборка промпта
            prompt = self.build_prompt(query, chunks, trace)
            # 3. Генерация ответа
            llm_result = self.generate_answer(prompt, trace)
            
            if not llm_result['success']:
                result = {
                    'query': query,
                    'answer': f"Ошибка при генерации ответа: {llm_result.get('error', 'Unknown error')}",
                    'sources': [],
                    'success': False
                }
                trace.update(output=result, level="ERROR")
                return result
            
            # 4. Форматирование цитат
            citations = self.format_citations(chunks)
            # 5. Финальный результат
            result = {
                'query': query,
                'answer': llm_result['answer'],
                'sources': citations,
                'metadata': {
                    'num_sources': len(chunks),
                    'model': llm_result['model'],
                    'tokens': llm_result['tokens'],
                    'timestamp': datetime.now().isoformat()
                },
                'success': True
            }
            
            # Завершаем trace с финальным результатом
            trace.update(
                output={
                    "answer": result['answer'],
                    "num_sources": result['metadata']['num_sources'],
                    "tokens": result['metadata']['tokens']
                },
                metadata={
                    "timestamp": result['metadata']['timestamp']
                }
            )
            
            return result


def print_result(result):
    """Красиво выводит результат RAG пайплайна."""
    print("\n" + "="*80)
    print(f"ВОПРОС: {result['query']}")
    print("="*80)
    
    if not result['success']:
        print(f"\n❌ {result['answer']}")
        return
    
    print(f"\n📝 ОТВЕТ:\n")
    print(result['answer'])
    print(f"\n\n📚 ИСТОЧНИКИ ({result['metadata']['num_sources']}):\n")
    print(result['sources'])
    print("\n" + "-"*80)
    metadata = result['metadata']
    print(f"Модель: {metadata['model']}")

@click.command()
@click.argument('question', required=True)
@click.option('--top_k', '-k', type=int, default=None, help='Количество контекстных документов для поиска')
@click.option('--config', '-c', default='./data/config.yaml', help='Путь к файлу конфигурации')
@click.option('--output', '-o', type=click.Path(), help='Сохранить результат в JSON файл')
@click.option('--session_id', '-s', type=str, default=None, help='Идентификатор сессии для отслеживания в Langfuse (если не указан, генерируется новый)')
def main(question, top_k, config, output, session_id):
    # Генерируем session_id один раз при старте CLI, если не передан
    if session_id is None:
        session_id = str(uuid.uuid4())
        print(f"🔑 Создана новая сессия: {session_id}")
    else:
        print(f"🔑 Используется сессия: {session_id}")
    
    rag = RAGPipeline(config_path=config)
    # Обработка вопроса
    result = rag.answer_question(question, top_k, session_id=session_id)
    print_result(result)

if __name__ == "__main__":
    main()