from typing import List, Dict, Any, Set
import numpy as np


class RetrievalMetrics:
    @staticmethod
    def recall_at_k(retrieved, relevant, k):
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        hits = retrieved_at_k & relevant
        
        return len(hits) / len(relevant)
    
    @staticmethod
    def precision_at_k(retrieved, relevant, k):

        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        hits = retrieved_at_k & relevant
        
        return len(hits) / k
    
    @staticmethod
    def mrr(retrieved, relevant):
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / rank
        return 0.0
    
    
    @staticmethod
    def compute_all_metrics(retrieved, relevant, k_values: List[int] = [5, 10]):
        metrics = {}
        
        for k in k_values:
            metrics[f'recall@{k}'] = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
            metrics[f'precision@{k}'] = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        
        metrics['mrr'] = RetrievalMetrics.mrr(retrieved, relevant)
        
        return metrics


class EvaluationAggregator:
    """Класс для агрегирования метрик по нескольким запросам."""
    
    def __init__(self):
        self.query_metrics = []
    
    def add_query_metrics(self, metrics: Dict[str, float]):
        """Добавляет метрики для одного запроса."""
        self.query_metrics.append(metrics)
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """
        Вычисляет средние значения метрик по всем запросам.
        
        Returns:
            Словарь со средними метриками
        """
        if not self.query_metrics:
            return {}
        
        # Собираем все ключи метрик
        all_keys = set()
        for metrics in self.query_metrics:
            all_keys.update(metrics.keys())
        
        # Вычисляем средние
        aggregated = {}
        for key in all_keys:
            values = [m.get(key, 0.0) for m in self.query_metrics]
            aggregated[key] = np.mean(values)
        
        return aggregated
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Возвращает полную сводку с метриками.
        
        Returns:
            Словарь со сводкой
        """
        aggregated = self.get_aggregated_metrics()
        
        return {
            'total_queries': len(self.query_metrics),
            'mean_metrics': aggregated,
            'per_query_metrics': self.query_metrics
        }
    
    def print_summary(self, detailed: bool = False):
        """
        Печатает красивую сводку метрик.
        
        Args:
            detailed: Показать детальные метрики по каждому запросу
        """
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("Метрики поиска:")
        print("=" * 80)
        print(f"Всего запросов: {summary['total_queries']}")
        print()
        
        # Средние метрики
        print("Средние метрики:")
        print("-" * 80)
        
        mean_metrics = summary['mean_metrics']
        
        # Группируем по типу метрики
        metric_groups = {}
        for key, value in mean_metrics.items():
            if '@' in key:
                base_metric, k = key.rsplit('@', 1)
                if base_metric not in metric_groups:
                    metric_groups[base_metric] = {}
                metric_groups[base_metric][k] = value
            else:
                metric_groups[key] = {'': value}
        
        for metric_name, values in sorted(metric_groups.items()):
            if isinstance(values, dict) and '' in values:
                # Метрика без k (например, MRR)
                print(f"  {metric_name.upper():15s}: {values['']:.2f}")
            else:
                # Метрики с разными k
                print(f"  {metric_name.upper():15s}:", end='')
                for k, value in sorted(values.items(), key=lambda x: int(x[0])):
                    print(f"  @{k}={value:.2f}", end='')
                print()
        
        print("=" * 80)
        
        if detailed and summary['per_query_metrics']:
            print("\nДетальные метрики по запросам:")
            print("-" * 80)
            for i, metrics in enumerate(summary['per_query_metrics'], 1):
                print(f"\nЗапрос {i}:")
                for key, value in sorted(metrics.items()):
                    print(f"  {key:20s}: {value:.2f}")
            print("=" * 80)