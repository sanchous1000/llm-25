from langfuse import Langfuse
from dotenv import load_dotenv
import os
import argparse

from rag_pipeline import run_pipeline

load_dotenv()


def count_for_metrics(indices, chunk_arxiv_ids, relevant_ids):
    retrieved_ids = [chunk_arxiv_ids[i] for i in indices[0]]

    relevant_retrieved = set([rid for rid in retrieved_ids if rid in relevant_ids])
    num_relevant_total = len(relevant_ids)
    num_retrieved = len(retrieved_ids)
    num_relevant_retrieved = len(relevant_retrieved)

    return num_relevant_total, num_retrieved, num_relevant_retrieved, retrieved_ids


def mmr(indices, chunk_arxiv_ids, relevant_ids):
    num_relevant_total, num_retrieved, num_relevant_retrieved, retrieved_ids = count_for_metrics(
        indices,
        chunk_arxiv_ids,
        relevant_ids
    )

    mrr = 0.0
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            mrr = 1.0 / rank
            break
    return mrr


def precession(indices, chunk_arxiv_ids, relevant_ids):
    num_relevant_total, num_retrieved, num_relevant_retrieved, retrieved_ids = count_for_metrics(
        indices,
        chunk_arxiv_ids,
        relevant_ids
    )
    return num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0


def recall(indices, chunk_arxiv_ids, relevant_ids):
    num_relevant_total, num_retrieved, num_relevant_retrieved, retrieved_ids = count_for_metrics(
        indices,
        chunk_arxiv_ids,
        relevant_ids
    )

    return num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0


def main(name_exp='index_5790b8cf'):
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )

    dataset = langfuse.get_dataset("arxiv-dataset")

    for item in dataset.items:
        with item.run(
            run_name=name_exp,
            run_description="Тестирование RAG на релевантность извлечения чанков",
        ) as root_span:
            output, chunk_arxiv_ids, indices = run_pipeline(item.input, langfuse, name_exp)

            root_span.score_trace(
                name="MMR",
                value=mmr(indices, chunk_arxiv_ids, item.metadata[0]),
            )
            root_span.score_trace(
                name="precession",
                value=precession(indices, chunk_arxiv_ids, item.metadata[0]),
            )
            root_span.score_trace(
                name="recall",
                value=recall(indices, chunk_arxiv_ids, item.metadata[0]),
            )

            langfuse.update_current_trace(input=item.input, output=output)

    langfuse.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rag', type=str, required=True)

    args = parser.parse_args()

    main(args.rag)
