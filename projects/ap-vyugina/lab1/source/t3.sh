PROMPT_3="Few-shot object detection aims at learning novel categories given only a few example images. It is a basic skill for a robot that performs tasks in open environments. Recent methods focus on finetuning strategies, with complicated procedures that prohibit a wider application. In this paper, we introduce DE-ViT, a few-shot object detector without the need for finetuning. DE-ViT ’s novel architecture is based on a new region-propagation mechanism for localization. The propagated region masks are transformed into bounding boxes through a learnable spatial integral layer. Instead of training prototype classifiers, we propose to use prototypes to project ViT features into a subspace that is robust to overfitting on base classes. We evaluate DE-ViT on few-shot, and one-shot object detection benchmarks with Pascal VOC, COCO, and LVIS. DE-ViT establishes new stateof-the-art results on all benchmarks. Notably, for COCO, DE-ViT surpasses the few-shot SoTA by 15 mAP on 10-shot and 7.2 mAP on 30-shot and one-shot SoTA by 2.8 AP50. For LVIS, DE-ViT outperforms few-shot SoTA by 17 box APr. Further, we evaluate DE-ViT with a real robot by building a pick-and-place system for sorting novel objects based on example images."

SYS_PROMPT="You are an expert in Deep learning. Provide a summary of the following abstract in one-two sentences, without any additional comments."

### DEFAULT PARAMS
llama=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "llama3.2:1b",
    "messages": [
        {"role": "system", "content": "'"$SYS_PROMPT"'"},
        {"role": "user", "content": "'"$PROMPT_3"'"}
    ],
    "stream": false
    }' localhost:11434/api/chat)

gemma=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "gemma3:270m",
    "messages": [
        {"role": "system", "content": "'"$SYS_PROMPT"'"},
        {"role": "user", "content": "'"$PROMPT_3"'"}
    ],
    "stream": false
    }' localhost:11434/api/chat)

qwen=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "qwen2.5:0.5b",
    "messages": [
        {"role": "system", "content": "'"$SYS_PROMPT"'"},
        {"role": "user", "content": "'"$PROMPT_3"'"}
    ],
    "stream": false
    }' localhost:11434/api/chat)

echo $llama | jq -r '[.model, "default", .message.content] | @csv' >> res3.csv
echo $gemma | jq -r '[.model, "default", .message.content] | @csv' >> res3.csv
echo $qwen  | jq -r '[.model, "default", .message.content] | @csv' >> res3.csv

echo ''
jq -r -n --arg v1 "$llama" --arg v2 "$gemma" --arg v3 "$qwen" '([$v1, $v2, $v3].[] | fromjson | ["task3", .model, "default", .total_duration, .load_duration, .prompt_eval_count, .prompt_eval_duration, .eval_count, .eval_duration]) |  @csv' >> stats.csv

### FINETUNED PARAMS
llama=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "llama3.2:1b",
    "messages": [
        {"role": "system", "content": "'"$SYS_PROMPT"'"},
        {"role": "user", "content": "'"$PROMPT_3"'"}
    ],
    "stream": false,
    "options": {"temperature": 0.2, "top_k":100}
    }' localhost:11434/api/chat)

gemma=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "gemma3:270m",
    "messages": [
        {"role": "system", "content": "'"$SYS_PROMPT"'"},
        {"role": "user", "content": "'"$PROMPT_3"'"}
    ],
    "stream": false,
    "options": {"temperature": 0.2, "top_k":100}
    }' localhost:11434/api/chat)

qwen=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "qwen2.5:0.5b",
    "messages": [
        {"role": "system", "content": "'"$SYS_PROMPT"'"},
        {"role": "user", "content": "'"$PROMPT_3"'"}
    ],
    "stream": false,
    "options": {"temperature": 0.2, "top_k":100}
    }' localhost:11434/api/chat)

echo $llama | jq -r '[.model, "finetuned", .message.content] | @csv' >> res3.csv
echo $gemma | jq -r '[.model, "finetuned", .message.content] | @csv' >> res3.csv
echo $qwen  | jq -r '[.model, "finetuned", .message.content] | @csv' >> res3.csv

echo ''
jq -r -n --arg v1 "$llama" --arg v2 "$gemma" --arg v3 "$qwen" '([$v1, $v2, $v3].[] | fromjson | ["task3", .model, "finetuned", .total_duration, .load_duration, .prompt_eval_count, .prompt_eval_duration, .eval_count, .eval_duration]) |  @csv' >> stats.csv
