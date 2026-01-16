# Лаба 2 — Построение RAG-агента по документации

### Выбрать и подготовить источники данных

Выбран репозиторий GPU Glossary https://github.com/modal-labs/gpu-glossary

Репрезентативные вопросы находятся в evaluation_set.json

### Спарсить источники в Markdown

Был реализован парсинг MD-файлов из репозитория. Код парсера находится в файле parse.py

Результаты парсинга находятся в папке /docs.

### Разбить на чанки и построить эмбеддинги (sparse/dense)

Реализовано разбиение на чанки и построение эмбеддингов с возможностью конфигурации параметров с помощью аргументов командной строки и повторного запуска.

Код находится в embedder.py.

### Развернуть векторное хранилище и загрузить индекс

Развернут FAISS. Реализован выбор параметров, а также возможность пересборки и повторных запусков.

Код находится в load_to_vector_store.py

### Метрики и оценка качества retrieval/QA

Оцениваются precision, recall, MRR.

Лучший результат:

```
Precision@5: 0.2133
Recall@5: 0.8667
Precision@10: 0.1200
Recall@10: 0.9333
MRR: 0.9429
```

Лучшие параметры: chunk_size=500, chunk_overlap=50

### Реализовать движок общения (RAG-пайплайн)

Реализован RAG-пайплайн. Пример работы:

```
User: What is Tensor core? 

Assistant:
Tensor Cores are GPU cores that operate on entire matrices with each instruction. They are much larger and less numerous than CUDA Cores, with an H100 SXM5 having only four Tensor Cores per Streaming Multiprocessor (SM), and they serve as the primary producers and consumers of Tensor Memory [Doc 1, Doc 2]. Tensor Cores were introduced in the V100 GPU, enhancing NVIDIA GPUs' suitability for large neural network workloads [Doc 3]. The internals of Tensor Cores are unknown and likely differ from SM Architecture [Doc 3].

[Doc 1], [Doc 2], [Doc 3]

Sources:
- [1] gpu-glossary/device-hardware/tensor-core.md | Root
- [2] gpu-glossary/device-hardware/tensor-core.md | Root
- [3] gpu-glossary/device-hardware/tensor-core.md | Root
- [4] gpu-glossary/device-hardware/tensor-memory.md | Root
```
