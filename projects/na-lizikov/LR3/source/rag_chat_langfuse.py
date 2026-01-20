import requests
import os
import sys
from dotenv import load_dotenv
from langfuse import Langfuse
import time

# Загружаем переменные окружения
load_dotenv()

# Добавляем путь к scripts для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LR2", "scripts"))
from retrieve import hybrid_search

# Настройки из переменных окружения
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Инициализация Langfuse (V2 API)
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)


def call_ollama(prompt, model=MODEL, trace_id=None):
    """Вызов Ollama с логированием в Langfuse V2"""
    start_time = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        answer = result["response"]

        # Логируем генерацию (V2 API)
        langfuse.generation(
            trace_id=trace_id,
            name="llm_generation",
            model=model,
            model_parameters={
                "temperature": 0.7,
                "max_tokens": 2048
            },
            input=prompt,
            output=answer,
            usage={
                "promptTokens": len(prompt.split()),
                "completionTokens": len(answer.split()),
                "totalTokens": len(prompt.split()) + len(answer.split())
            },
            metadata={
                "duration_sec": time.time() - start_time
            }
        )

        return answer

    except requests.exceptions.RequestException as e:
        error_msg = f"Ошибка при обращении к Ollama: {e}"

        langfuse.generation(
            trace_id=trace_id,
            name="llm_generation_error",
            model=model,
            input=prompt,
            output=error_msg,
            metadata={"error": True}
        )

        return error_msg


def retrieve_chunks(question, top_k=5, trace_id=None):
    """Поиск релевантных чанков с логированием (V2 API)"""

    retrieval_span = None
    if trace_id:
        try:
            retrieval_span = langfuse.span(
                trace_id=trace_id,
                name="retrieval",
                input=question,
                metadata={
                    "top_k": top_k,
                    "method": "hybrid_search"
                }
            )
        except Exception:
            # Если не удалось создать span, продолжаем без логирования
            pass

    hits = hybrid_search(question, top_k)

    if not hits:
        if retrieval_span:
            try:
                retrieval_span.update(
                    output={"status": "no_results", "hits": []}
                )
            except Exception:
                pass
        return None, []

    contexts = []
    sources = []

    for i, h in enumerate(hits, 1):
        src = h["_source"]
        score = h.get("_score", 0.0)

        text = src.get("text", "")
        snippet = text[:500].replace("\n", " ").strip()
        if len(text) > 500:
            snippet += "..."

        source = src.get("source", "Неизвестный источник")
        relative_path = src.get("relative_path", "")

        contexts.append(
            f"[Контекст {i}] (релевантность: {score:.2f})\n"
            f"Источник: {source}\n"
            f"Путь: {relative_path}\n"
            f"Текст: {snippet}\n"
        )

        sources.append({
            "source": source,
            "path": relative_path,
            "score": float(score),
            "text": snippet
        })

    if retrieval_span:
        try:
            retrieval_span.update(
                output={
                    "status": "success",
                    "num_hits": len(hits),
                    "sources": sources
                }
            )
        except Exception:
            pass

    return contexts, sources


def ask(question, top_k=5, user_id=None):
    """RAG-пайплайн с полным логированием в Langfuse V2"""
    trace = None
    trace_id = None
    
    # В V2 создаем trace явно через trace() метод
    try:
        trace = langfuse.trace(
            name="rag_chat",
            user_id=user_id,
            input={"question": question},
            metadata={
                "top_k": top_k,
                "question_preview": question[:100]
            }
        )
        # Получаем trace_id из объекта trace
        trace_id = trace.id if hasattr(trace, 'id') else None
        if not trace_id:
            # Если id недоступен напрямую, пробуем получить через метод
            try:
                trace_id = trace.get_trace_id() if hasattr(trace, 'get_trace_id') else None
            except:
                pass
    except Exception as e:
        print(f"⚠ Не удалось создать trace в Langfuse: {e}")
        import traceback
        traceback.print_exc()
        trace_id = None
        trace = None

    try:
        # 1. Retrieval
        contexts, sources = retrieve_chunks(question, top_k, trace_id=trace_id)

        if not contexts:
            if trace:
                try:
                    trace.update(output="Релевантная информация не найдена.")
                except Exception:
                    pass
            return "Извините, не удалось найти релевантную информацию в документации."

        # 2. Prompt
        prompt = f"""Ты — ассистент, отвечающий строго по документации Python.
Используй ТОЛЬКО информацию из предоставленного контекста.
Если ответа нет — честно скажи об этом.

Контекст:
{chr(10).join(contexts)}

Вопрос:
{question}

Требования:
- Используй только контекст
- В конце укажи источники (источник и путь)

Ответ:
"""

        # 3. LLM
        answer = call_ollama(prompt, trace_id=trace_id)

        # 4. Финальный ответ
        sources_text = "\n\nИсточники:\n"
        for src in sources:
            sources_text += f"- {src['source']} ({src['path']})\n"

        final_answer = answer + sources_text

        if trace:
            try:
                trace.update(
                    output=final_answer,
                    metadata={
                        "num_sources": len(sources),
                        "source_paths": [s["path"] for s in sources]
                    }
                )
            except Exception as e:
                print(f"⚠ Ошибка обновления trace: {e}")
        
        # Отправляем данные в Langfuse
        try:
            langfuse.flush()
            if trace_id:
                trace_id_str = str(trace_id)
                print(f"✓ Данные отправлены в Langfuse (trace_id: {trace_id_str[:8]}...)")
        except Exception as e:
            print(f"⚠ Ошибка отправки данных в Langfuse: {e}")

        return final_answer

    except Exception as e:
        if trace:
            try:
                trace.update(output=str(e), metadata={"error": True})
            except Exception:
                pass
        # Отправляем данные даже при ошибке
        try:
            langfuse.flush()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    print("RAG-чат с Langfuse запущен. Введите 'exit' для выхода.\n")
    user_id = input("Введите ваш ID пользователя (или Enter для анонимного): ").strip() or None

    while True:
        try:
            q = input("\nВопрос: ").strip()
            if q.lower() in ["exit", "quit", "выход"]:
                print("До свидания!")
                langfuse.flush()
                break

            if not q:
                continue

            print("\n" + "=" * 50)
            answer = ask(q, user_id=user_id)
            print(answer)
            print("=" * 50)

        except KeyboardInterrupt:
            print("\nДо свидания!")
            langfuse.flush()
            break

        except Exception as e:
            error_msg = str(e)
            print(f"Ошибка: {error_msg}\n")
            
            # Если ошибка связана с Langfuse, продолжаем работу без логирования
            if "langfuse" in error_msg.lower() or "bad gateway" in error_msg.lower() or "api error" in error_msg.lower():
                print("⚠ Предупреждение: Проблема с логированием в Langfuse. Продолжаем работу...\n")
                try:
                    # Пытаемся выполнить без логирования
                    from LR2.scripts.retrieve import hybrid_search
                    hits = hybrid_search(q, 5)
                    if hits:
                        contexts = []
                        for h in hits[:5]:
                            src = h["_source"]
                            text = src.get("text", "")[:500]
                            contexts.append(f"{text}...")
                        prompt = f"""Ответь на вопрос используя контекст:
{chr(10).join(contexts)}

Вопрос: {q}
Ответ:"""
                        import requests
                        response = requests.post(
                            OLLAMA_URL,
                            json={"model": MODEL, "prompt": prompt, "stream": False},
                            timeout=120
                        )
                        answer = response.json()["response"]
                        print(answer)
                except:
                    pass
