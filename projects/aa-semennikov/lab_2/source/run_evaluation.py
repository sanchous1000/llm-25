import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import click
sys.path.insert(0, str(Path(__file__).parent.parent))
from source.utils import load_config
from source.embeddings import DenseEmbedder
from source.evaluation_metrics import RetrievalMetrics, EvaluationAggregator

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("Ошибка: qdrant-client не установлен")
    sys.exit(1)


class RetrievalEvaluator:
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = load_config(config_path)
        qdrant_config = self.config.get('qdrant', {})
        self.host = qdrant_config.get('host', 'localhost')
        self.port = qdrant_config.get('port', 6333)
        self.collection_name = qdrant_config.get('collection_name', 'documents')
        self.ef_search = qdrant_config.get('hnsw', {}).get('ef_search', 100)
            
        self.client = QdrantClient(host=self.host, port=self.port)
        self.embedder = DenseEmbedder(self.config['embeddings']['dense'])
    
    def search(self, query, limit = 20):
        """
        Выполняет поиск и возвращает список документов.
        
        Args:
            query: Текст запроса
            limit: Максимальное количество результатов
            
        Returns:
            Список имен документов
        """
        model_name = self.config['embeddings']['dense']['model'].lower()
        
        if 'bge' in model_name:
            query_text = f"Represent this sentence for searching relevant passages: {query}"
        elif 'e5' in model_name:
            query_text = f"query: {query}"
        else:
            query_text = query
        
        embeddings_result = self.embedder.embed_texts([query_text])
        query_vector = embeddings_result['dense'][0].tolist()
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        ).points
        
        # Извлекаем имена документов
        retrieved_docs = [result.payload.get('document', '') for result in results]
        
        return retrieved_docs
    
    def evaluate_query(self, query, relevant_docs, k_values = [5, 10], max_results = 20):
        start_time = time.time()
        
        retrieved_docs = self.search(query, limit=max_results)
        
        relevant_set = set(relevant_docs)
        metrics = RetrievalMetrics.compute_all_metrics(retrieved_docs, relevant_set, k_values)
        
        elapsed_time = time.time() - start_time
        
        return {
            'query': query,
            'retrieved': retrieved_docs,
            'relevant': relevant_docs,
            'metrics': metrics,
            'query_time': elapsed_time
        }
    
    def evaluate_queries(self, queries_file, k_values = [5, 10], max_results = 20):
        """
        Оценивает набор запросов из файла.
        
        Args:
            queries_file: Путь к JSON файлу с запросами
            k_values: Значения k для метрик
            max_results: Максимальное количество результатов
            
        Returns:
            Полный отчет с результатами
        """
        total_start_time = time.time()
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        queries = queries_data.get('queries', [])
        # Агрегатор метрик
        aggregator = EvaluationAggregator()
        # Результаты по запросам
        query_results = []
        
        # Оцениваем каждый запрос
        for i, query_data in enumerate(queries, 1):
            query_text = query_data['query']
            relevant_docs = query_data.get('relevant_documents', [])
            
            try:
                result = self.evaluate_query(
                    query=query_text,
                    relevant_docs=relevant_docs,
                    k_values=k_values,
                    max_results=max_results
                )
                
                # Добавляем ID и описание
                result['id'] = query_data.get('id', i)
                result['description'] = query_data.get('description', '')
                
                query_results.append(result)
                aggregator.add_query_metrics(result['metrics'])
            
            except Exception as e:
                continue
        
        # Агрегированные метрики
        summary = aggregator.get_summary()
        
        total_elapsed_time = time.time() - total_start_time
        avg_query_time = total_elapsed_time / len(query_results) if query_results else 0
        
        # Формируем полный отчет
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'evaluation_params': {
                'k_values': k_values,
                'max_results': max_results,
                'queries_file': queries_file
            },
            'timing': {
                'total_time': total_elapsed_time,
                'avg_query_time': avg_query_time,
                'queries_count': len(query_results)
            },
            'summary': summary,
            'query_results': query_results
        }
        
        return report
    
    def print_report(self, report, detailed = False):
        print("\n" + "=" * 80)
        print("ОТЧЕТ EVALUATION")
        print("=" * 80)
        print("\nКонфигурация:")
        print(f"  Модель эмбеддингов: {report['config']['embeddings']['dense']['model']}")
        print(f"  Chunk size: {report['config']['chunking']['chunk_size']}")
        print(f"  Chunk overlap: {report['config']['chunking']['chunk_overlap']}")
        print(f"  HNSW M: {report['config']['qdrant']['hnsw']['m']}")
        print(f"  HNSW ef_construction: {report['config']['qdrant']['hnsw']['ef_construction']}")
        params = report['evaluation_params']
        print(f"\nПараметры оценки:")
        print(f"  K values: {params['k_values']}")
        print(f"  Max results: {params['max_results']}")
        
        if 'timing' in report:
            timing = report['timing']
            print(f"\nВремя выполнения:")
            print(f"  Общее время: {timing['total_time']:.2f} секунд")
            print(f"  Среднее время на запрос: {timing['avg_query_time']:.3f} секунд")
            print(f"  Обработано запросов: {timing['queries_count']}")
        
        summary = report['summary']
        print(f"\n" + "=" * 80)
        print(f"СВОДКА МЕТРИК ({summary['total_queries']} запросов)")
        print("=" * 80)
        
        mean_metrics = summary['mean_metrics']
        
        for k in params['k_values']:
            print(f"\nМетрики @{k}:")
            print(f"  Recall@{k}:    {mean_metrics.get(f'recall@{k}', 0):.2f}")
            print(f"  Precision@{k}: {mean_metrics.get(f'precision@{k}', 0):.2f}")
        
        print(f"\nMRR: {mean_metrics.get('mrr', 0):.2f}")
        
        print("=" * 80)
        
        # Детальные результаты
        if detailed:
            print("\nДетальные результаты по запросам:")
            print("-" * 80)
            
            for result in report['query_results']:
                print(f"\n[Query {result['id']}] {result['query']}")
                print(f"Релевантные: {result['relevant']}")
                print(f"Найдены в топ-5:")
                
                top5 = result['retrieved'][:5]
                for i, doc in enumerate(top5, 1):
                    marker = "✓" if doc in result['relevant'] else " "
                    print(f"  {i}. [{marker}] {doc}")
                
                metrics = result['metrics']
                print(f"Метрики: Recall@5={metrics['recall@5']:.2f}, "
                     f"Precision@5={metrics['precision@5']:.2f}, "
                     f"MRR={metrics['mrr']:.2f}")
            
            print("=" * 80)


@click.command()
@click.option('--config', default='config.yaml', help='Путь к файлу конфигурации')
@click.option('--queries', default='evaluation_queries.json', help='Путь к JSON файлу с запросами для оценки')
@click.option('--k', multiple=True, type=int, help='Значения k для метрик (можно указать несколько раз, например --k 5 --k 10)')
@click.option('--max-results', default=20, type=int, help='Максимальное количество результатов поиска')
@click.option('--output', help='Путь для сохранения отчета в JSON формате')
@click.option('--detailed', is_flag=True, help='Вывести детальные результаты по каждому запросу')
def main(config, queries, k, max_results, output, detailed):
    
    script_start_time = time.time()
    
    try:
        if not Path(queries).exists():
            print(f"Ошибка: файл с запросами не найден: {queries}")
            sys.exit(1)
            
        evaluator = RetrievalEvaluator(config_path=config)
        k_values = list(k) if k else [5, 10]
        
        print(f"\nЗапуск evaluation...")
        print(f"  Запросы: {queries}")
        print(f"  K values: {k_values}")
        print(f"  Max results: {max_results}")
        print()
        
        report = evaluator.evaluate_queries(
            queries_file=queries,
            k_values=k_values,
            max_results=max_results
        )
        
        # Выводим отчет
        evaluator.print_report(report, detailed=detailed)
        
        # Сохраняем если указан output
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Отчет сохранен: {output_path}")
        
        script_elapsed_time = time.time() - script_start_time
        print(f"\n✓ Скрипт завершен за {script_elapsed_time:.2f} секунд ({script_elapsed_time/60:.2f} минут)")
    
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()