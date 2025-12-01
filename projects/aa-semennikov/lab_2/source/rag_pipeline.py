import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import click
import time
sys.path.insert(0, str(Path(__file__).parent.parent))
from source.utils import load_config
from source.embeddings import DenseEmbedder
from qdrant_client import QdrantClient
from openai import OpenAI


class RAGPipeline:
    
    def __init__(self, config_path = 'config.yaml'):
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
    
    def search_relevant_chunks(self, query, top_k = None):
        """
        Поиск релевантных чанков в векторном хранилище.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов (если не указано, используется из конфига)
            
        Returns:
            Список словарей с информацией о чанках и их релевантности
        """
        if top_k is None:
            top_k = self.top_k
        
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
        
        return chunks
    
    def build_prompt(self, query, chunks):
        """
        Собирает промпт для LLM: инструкции + вопрос + контексты.
        
        Args:
            query: Вопрос пользователя
            chunks: Список релевантных чанков
            
        Returns:
            Готовый промпт для LLM
        """
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
        
        return prompt
    
    def generate_answer(self, prompt):
        """
        Генерирует ответ с помощью LLM.
        
        Args:
            prompt: Промпт для LLM
            
        Returns:
            Словарь с ответом и метаинформацией
        """
        start_time = time.time()
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            end_time = time.time()
            
            usage = response.usage
            answer_text = response.choices[0].message.content.strip()
            
            return {
                'success': True,
                'answer': answer_text,
                'model': self.llm_model,
                'tokens': {
                    'prompt': usage.prompt_tokens,
                    'completion': usage.completion_tokens,
                    'total': usage.total_tokens
                },
                'time_sec': round(end_time - start_time, 2)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': None
            }
    
    def format_citations(self, chunks):
        """
        Форматирует цитаты (источники) для ответа.
        
        Args:
            chunks: Список использованных чанков
            
        Returns:
            Отформатированная строка с источниками
        """
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
    
    def answer_question(self, query, top_k = None):
        """
        Полный RAG пайплайн: поиск + генерация + форматирование ответа.
        
        Args:
            query: Вопрос пользователя
            top_k: Количество контекстов (если не указано, используется из конфига)
            
        Returns:
            Словарь с полным ответом, цитатами и метаинформацией
        """
        # 1. Поиск релевантных чанков
        chunks = self.search_relevant_chunks(query, top_k)
        
        if not chunks:
            return {
                'query': query,
                'answer': "К сожалению, я не нашел релевантных документов для ответа на ваш вопрос.",
                'sources': [],
                'success': False
            }
        
        # 2. Сборка промпта
        prompt = self.build_prompt(query, chunks)
        
        # 3. Генерация ответа
        llm_result = self.generate_answer(prompt)
        
        if not llm_result['success']:
            return {
                'query': query,
                'answer': f"Ошибка при генерации ответа: {llm_result.get('error', 'Unknown error')}",
                'sources': [],
                'success': False
            }
        
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
                'generation_time_sec': llm_result['time_sec'],
                'timestamp': datetime.now().isoformat()
            },
            'success': True
        }
        
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
    print(f"Токены: {metadata['tokens']['prompt']} (промпт) + {metadata['tokens']['completion']} (ответ) = {metadata['tokens']['total']} (всего)")
    print(f"Время генерации: {metadata['generation_time_sec']} сек")
    print("="*80 + "\n")


@click.command()
@click.argument('question', required=True)
@click.option('--top_k', '-k', type=int, default=None, help='Количество контекстных документов для поиска')
@click.option('--config', '-c', default='config.yaml', help='Путь к файлу конфигурации')
@click.option('--output', '-o', type=click.Path(), help='Сохранить результат в JSON файл')
def main(question, top_k, config, output):
    """
    RAG Pipeline для ответов на вопросы с использованием векторного поиска и LLM.
    
    Примеры использования:
    
        # Базовое использование
        python scripts/rag_pipeline.py "Что такое машинное обучение?"
        
        # С указанием количества источников
        python scripts/rag_pipeline.py "Что такое машинное обучение?" --top_k 3
        
        # Сохранение результата
        python scripts/rag_pipeline.py "Вопрос?" --output result.json
    """
    # Инициализация RAG Pipeline
    try:
        rag = RAGPipeline(config_path=config)
    except Exception as e:
        print(f"❌ Ошибка при инициализации RAG Pipeline: {e}")
        sys.exit(1)
    
    # Обработка вопроса
    result = rag.answer_question(question, top_k)
    print_result(result)
    
    # Сохранение в файл
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Результат сохранен в {output_path}")


if __name__ == '__main__':
    main()