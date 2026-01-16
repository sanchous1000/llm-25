import time
from typing import Dict, Any, Optional
from langfuse.openai import OpenAI
from langfuse import Langfuse

class LLMClient:
    def __init__(
        self, 
        base_url: str = "http://localhost:8080/v1", 
        api_key: str = "sk-no-key-required"
        , lf: Optional[Langfuse] = None
        ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        try:
            self.langfuse = lf if lf is not None else Langfuse()
        except Exception:
            try:
                self.langfuse = Langfuse()
            except Exception:
                self.langfuse = None
        
    def request(
        self, 
        model_name: str, 
        prompt: str, 
        system_prompt: str = "You are a helpful assistant.",
        **kwargs
    ) -> Dict[str, Any]:
        
        lf_name = kwargs.pop("langfuse_name", f"run-{int(time.time())}")
        lf_metadata = kwargs.pop("langfuse_metadata", {})

        start_time = time.perf_counter()
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                **kwargs 
            )
            
            end_time = time.perf_counter()
            latency = end_time - start_time
            
            return {
                "text": response.choices[0].message.content,
                "latency_sec": round(latency, 3),
                "tokens_prompt": response.usage.prompt_tokens,
                "tokens_completion": response.usage.completion_tokens,
                "tokens_total": response.usage.total_tokens,
                "tokens_per_sec": round(response.usage.completion_tokens / latency, 2) if latency > 0 else 0,
                "model": model_name,
                "params": kwargs
            }
            
        except Exception as e:
            return {"error": str(e)}
        