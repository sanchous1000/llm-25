import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import click
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent.parent))
from langfuse import Langfuse
from source.rag_pipeline import RAGPipeline
from source.evaluation_metrics import RetrievalMetrics
from source.create_dataset import create_dataset

load_dotenv()


def get_or_create_dataset(langfuse: Langfuse, dataset_name: str, create_if_missing: bool = False):
    print(f"Получение датасета: {dataset_name}")
    
    try:
        dataset = langfuse.get_dataset(dataset_name)
        # Подсчитываем количество элементов, перебирая items как итератор
        items_count = sum(1 for _ in dataset.items)
        print(f"Датасет '{dataset_name}' найден ({items_count} элементов)")
        # Возвращаем датасет заново, так как items это итератор и уже исчерпан
        return langfuse.get_dataset(dataset_name)
    except Exception as e:
        if create_if_missing:
            print(f"Датасет '{dataset_name}' не найден. Создаем новый...")
            created_dataset = create_dataset()
            
            # После создания получаем датасет заново через API
            print(f"Получение созданного датасета через API...")
            time.sleep(1)  # Небольшая задержка для синхронизации с API
            
            dataset = langfuse.get_dataset(created_dataset.name)
            items_count = sum(1 for _ in dataset.items)
            print(f"Датасет '{created_dataset.name}' создан ({items_count} элементов)")
            
            # Возвращаем датасет заново
            return langfuse.get_dataset(created_dataset.name)
        else:
            print(f"Ошибка при получении датасета '{dataset_name}': {str(e)}")
            print("Используйте --create-dataset для автоматического создания датасета")
            raise


def run_rag_pipeline(
    item_input: Dict[str, Any],
    rag_pipeline: RAGPipeline,
    top_k: int,
    trace
) -> tuple:
    query = item_input["question"]
    
    # 1. Retrieval - извлечение релевантных чанков
    chunks = rag_pipeline.search_relevant_chunks(query, top_k=top_k, trace=trace)
    retrieved_docs = [chunk['document'] for chunk in chunks]
    
    # 2. Сборка промпта
    prompt = rag_pipeline.build_prompt(query, chunks, trace=trace)
    
    # 3. Генерация ответа LLM
    generation_result = rag_pipeline.generate_answer(prompt, trace=trace)
    answer = generation_result.get('answer', 'N/A')
    
    return answer, retrieved_docs, chunks


def run_experiment(
    experiment_name: str,
    dataset_name: str = 'dataset_2',
    config_path: str = './data/config.yaml',
    top_k: int = 5,
    create_if_missing: bool = False
):
    print("\n" + "="*80)
    print("🚀 Запуск эксперимента оценки RAG-пайплайна")
    print("="*80)
    
    # Инициализируем Langfuse
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    
    # Получаем или создаем датасет
    dataset = get_or_create_dataset(langfuse, dataset_name, create_if_missing)
    
    # Инициализируем RAG pipeline
    print(f"\n🔧 Инициализация RAG pipeline...")
    rag_pipeline = RAGPipeline(config_path=config_path)
    print(f"RAG pipeline готов (model: {rag_pipeline.llm_model})")
    
    # Запускаем эксперимент
    print(f"\n📊 Запуск эксперимента: {experiment_name}")
    print(f"   Датасет: {dataset_name}")
    print(f"   Top-K: {top_k}")
    print("-"*80)
    
    # Получаем элементы датасета (dataset.items это итератор/генератор)
    dataset_items = list(dataset.items)
    total_items = len(dataset_items)
    
    print(f"\n🔄 Обработка {total_items} элементов датасета...\n")
    
    # Обрабатываем каждый элемент датасета
    for idx, item in enumerate(dataset_items, 1):
        query = item.input["question"]
        expected_docs = set(item.expected_output["relevant_documents"])
        
        print(f"[{idx}/{total_items}] 🔍 {query[:60]}...")
        
        # Создаем run для каждого элемента датасета
        with item.run(
            run_name=experiment_name,
            run_description="Оценка RAG пайплайна на релевантность извлечения чанков",
        ) as root_span:
            try:
                # Выполняем RAG pipeline
                answer, retrieved_docs, chunks = run_rag_pipeline(
                    item.input,
                    rag_pipeline,
                    top_k,
                    root_span
                )
                
                # Вычисляем метрики
                precision_at_k = RetrievalMetrics.precision_at_k(
                    retrieved=retrieved_docs,
                    relevant=expected_docs,
                    k=top_k
                )
                
                recall_at_k = RetrievalMetrics.recall_at_k(
                    retrieved=retrieved_docs,
                    relevant=expected_docs,
                    k=top_k
                )
                
                mrr = RetrievalMetrics.mrr(
                    retrieved=retrieved_docs,
                    relevant=expected_docs
                )
                
                # Добавляем метрики как scores через root_span.score_trace()
                root_span.score_trace(
                    name=f"precision@{top_k}",
                    value=precision_at_k,
                    comment=f"Precision@{top_k} for retrieval"
                )
                
                root_span.score_trace(
                    name=f"recall@{top_k}",
                    value=recall_at_k,
                    comment=f"Recall@{top_k} for retrieval"
                )
                
                root_span.score_trace(
                    name="mrr",
                    value=mrr,
                    comment="Mean Reciprocal Rank"
                )
                
                # Обновляем текущий trace с input и output
                langfuse.update_current_trace(
                    input=item.input,
                    output={
                        "answer": answer,
                        "retrieved_documents": retrieved_docs,
                        "num_chunks": len(chunks)
                    }
                )
                
                print(f"    Precision@{top_k}: {precision_at_k:.3f} | Recall@{top_k}: {recall_at_k:.3f} | MRR: {mrr:.3f}")
                
            except Exception as e:
                print(f"    Ошибка при обработке: {str(e)}")
                # Логируем ошибку в trace
                root_span.update(
                    level="ERROR",
                    status_message=str(e)
                )
    
    print(f"\nЭксперимент завершен!")
    print(f"   Обработано элементов: {total_items}")
    print("\n" + "="*80)
    
    # Flush для отправки всех данных в Langfuse
    langfuse.flush()


@click.command()
@click.option('--experiment-name', '-e', required=True, help='Название эксперимента')
@click.option('--dataset-name', '-d', default='dataset_2', help='Название датасета в Langfuse (по умолчанию: dataset_2)')
@click.option('--config', '-c', default='./data/config.yaml', help='Путь к конфигурации RAG')
@click.option('--top-k', '-k', type=int, default=5, help='Количество документов для retrieval')
@click.option('--create-dataset', is_flag=True, default=False, help='Создать датасет, если он не найден')
def main(experiment_name, dataset_name, config, top_k, create_dataset):
    run_experiment(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        config_path=config,
        top_k=top_k,
        create_if_missing=create_dataset
    )


if __name__ == "__main__":
    main()