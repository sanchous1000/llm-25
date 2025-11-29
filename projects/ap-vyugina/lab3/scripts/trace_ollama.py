import os

import requests
from dotenv import load_dotenv

load_dotenv()

import argparse
import time
import uuid

import langfuse
from langfuse import Langfuse


def request_ollama(prompt, model_name='llama'):
    base_url = f'http://localhost:11434/api/generate'

    data = {
        'model': model_name,
        'prompt': prompt,
        "stream": False
    }

    response = requests.post(base_url, json=data)

    if response.status_code != 200:
        raise ValueError(f' Ollama server returned an error {response.status_code}: {response.text}')

    return response.json()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="qwen3:1.7b")
    args = parser.parse_args()

    langfuseClient = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"), 
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        base_url=os.getenv("LANGFUSE_BASE_URL"),
    )

    with langfuse.propagate_attributes(
        user_id="avyuga",
        session_id=str(uuid.uuid4()),
        metadata={"environment": "test"}
    ):
        with langfuseClient.start_as_current_observation(
            as_type="generation", 
            name="process-query"
        ) as gen:
    
            start_time = time.time()
            response = request_ollama(args.query, args.model)
            end_time = time.time()

            gen.update(
                user_id="avyuga",
                model=args.model,
                input={"query": args.query},
                output=response,
                model_parameters={},
                usage_details={},
                cost_details={}
            )

    langfuseClient.flush()

            

