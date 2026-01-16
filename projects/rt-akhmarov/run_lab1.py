import os
import argparse
import pandas as pd
from langfuse import Langfuse
from src.llm import LLMClient
from src.utils import load_config, sync_prompts_to_langfuse
from dotenv import load_dotenv

load_dotenv()

def main(args):
    llm = LLMClient()
    lf = Langfuse()

    sync_prompts_to_langfuse(args.prompts, lf)
    modes_cfg = load_config(args.modes)["models"]
    task_inputs = load_config(args.inputs)
    
    current_model = os.getenv("LLM_MODEL_NAME", "unknown-model")
    results = []

    for p_name, inputs in task_inputs.items():
        try:
            prompt_reg = llm.langfuse.get_prompt(p_name)
            compiled_msgs = prompt_reg.compile(**inputs)
            
            sys_msg = next(m["content"] for m in compiled_msgs if m["role"] == "system")
            usr_msg = next(m["content"] for m in compiled_msgs if m["role"] == "user")

            for m_name, params in modes_cfg.items():
                print(f"Running: {current_model} | {p_name} | {m_name}")
                
                res = llm.request(
                    model_name=current_model,
                    prompt=usr_msg,
                    system_prompt=sys_msg,
                    langfuse_name=f"lab1-{current_model}-{p_name}-{m_name}",
                    langfuse_prompt=prompt_reg,
                    langfuse_metadata={"mode": m_name, "task": p_name},
                    **params
                )

                if "error" in res:
                    print(f"Error: {res['error']}")
                    continue

                results.append({
                    "model": current_model, "task": p_name, "mode": m_name,
                    "latency": res["latency_sec"], "tps": res["tokens_per_sec"],
                    "tokens": res["tokens_total"], "response": res["text"]
                })
        except Exception as e:
            print(f"Failed to process {p_name}: {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(f"{args.output_dir}/lab1_{current_model}.csv", index=False)
    lf.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="config/prompts.yml")
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--modes", type=str, default="config/models.yml")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)