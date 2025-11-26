import json
import argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

from utils import load_config

OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY = "pass"

MODES = {
    "basic": {"temperature": 0.8, "max_tokens": 128, "repeat_penalty": 1.1},
    "tuned": {"temperature": 0.3, "max_tokens": 256, "repeat_penalty": 1.3}
}

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)


def query_model_with_metrics(model: str, prompt: str, params: dict) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # temperature=params["temperature"],
            # max_tokens=params["max_tokens"],
            # extra_body={"repeat_penalty": params["repeat_penalty"]}
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
        }
    except Exception as e:
        return {
            "answer": f"[ОШИБКА: {str(e)}]",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)

    args = parser.parse_args()

    config = load_config()

    config_hash = '5790b8cf'
    artifacts_dir = Path(f"../artifacts/index_{config_hash}")
    eval_dir = Path(f"../eval")
    results_path = eval_dir / "results.json"

    if not artifacts_dir.exists():
        raise RuntimeError(
            f"Индекс не найден: {artifacts_dir}. Сначала запустите build_index.py и load_to_vector_store.py")

    print("Загрузка FAISS-индекса...")
    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    with open(artifacts_dir / "metadata.json", encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = SentenceTransformer(config["embedding"]["model"])
    top_k = config["retrieval"]["top_k"]

    if args.mode == 'test':
        with open(eval_dir / "questions.json", encoding="utf-8") as f:
            questions = json.load(f)
    else:
        questions = [{'question': str(input())}]

    results = []

    for q in questions:
        question = q['question']

        query_vec = embedder.encode(question)
        faiss.normalize_L2(query_vec.reshape(1, -1))

        distances, indices = index.search(query_vec.reshape(1, -1).astype(np.float32), k=top_k)

        contexts = []
        sources = []
        for idx in indices[0]:
            chunk = chunks[idx]
            text = chunk["text"]
            arxiv_id = chunk.get("arxiv_id", "unknown")
            page = chunk.get("page", "unknown")

            contexts.append(text)
            sources.append((arxiv_id, page))

        if not contexts:
            answer_text = "Не найдено релевантных фрагментов."
            print("Не найдено релевантных фрагментов.")
        else:
            context_block = "\n\n".join(
                f"[Источник: arXiv:{arxiv_id}, стр. {page}]\n{ctx}"
                for (arxiv_id, page), ctx in zip(sources, contexts)
            )

            prompt = f"""Ответь на вопрос, используя ТОЛЬКО приведённую ниже информацию.
                Не выдумывай. Если в тексте нет ответа — напиши "Информация отсутствует".

                Вопрос: {question}

                Контекст:
                {context_block}

                Ответ:
            """

            answer = query_model_with_metrics("mistral:7b-instruct-v0.3-q4_0", prompt, MODES['basic'])
            answer_text = answer['answer']

        # Сохраняем вопрос и ответ
        results.append({
            "question": question,
            "answer": answer_text,
            "sources": [{"arxiv_id": arxiv_id, "page": page} for arxiv_id, page in sources]
        })

        print("\n" + "=" * 60)
        print("Ответ:")
        print(answer_text)
        print("\n📚 Источники:")
        for (arxiv_id, page) in sources:
            print(f"- arXiv:{arxiv_id}, страница {page}")

    # Сохраняем все результаты в JSON
    eval_dir.mkdir(exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в {results_path}")


if __name__ == "__main__":
    main()