import time
from openai import OpenAI
from langfuse import propagate_attributes


OLLAMA_BASE_URL = "http://localhost:11434/v1"
API_KEY = "pass"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)


def query_model_with_metrics(model: str, prompt: str, langfuse, trace_id) -> str:
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage

    answer = response.choices[0].message.content.strip()
    end_time = time.time()

    delta_time = end_time - start_time

    with propagate_attributes(user_id="user_12345"):
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-call",
            input={"query": prompt},
            trace_context={"trace_id": trace_id},
            metadata={
                'time': delta_time,
                'model': model,
                'output_token': usage.completion_tokens,
                'input_token': usage.prompt_tokens,
                'total_token': usage.total_tokens
            }
        ) as root_span:
            root_span.update(output={"answer": answer})

    return answer
