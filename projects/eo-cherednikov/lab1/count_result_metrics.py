import json, collections
from pathlib import Path
from statistics import mean

results = [json.loads(l) for l in Path("results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

by_model = collections.defaultdict(list)
for r in results:
    model = r["model"]
    resp_text = ""
    try:
        choices = r["response"].get("choices")
        if choices and len(choices)>0:
            resp_text = choices[0].get("message", {}).get("content") or choices[0].get("text","")
    except Exception:
        resp_text = str(r["response"])
    r["__resp_text"] = resp_text
    r["__len_chars"] = len(resp_text)
    by_model[model].append(r)


for m, arr in by_model.items():
    times = [x["elapsed_s"] for x in arr]
    lens = [x["__len_chars"] for x in arr]
    print(f"MODEL {m}: runs={len(arr)}, avg_time_s={mean(times):.2f}, avg_len_chars={mean(lens):.1f}")