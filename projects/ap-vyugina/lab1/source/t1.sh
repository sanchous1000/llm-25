PROMPT="Generate a proverb about weather. Do not comment or explain meaning, return only the proverb in one sentence."

llama=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
    "stream": false
    }' localhost:11434/api/chat)

gemma=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "gemma3:270m",
    "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
    "stream": false
    }' localhost:11434/api/chat)

qwen=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "qwen2.5:0.5b",
    "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
    "stream": false
    }' localhost:11434/api/chat)

echo $llama | jq -r '[.model, "default", .message.content] | @csv' >> res1.csv
echo $gemma | jq -r '[.model, "default", .message.content] | @csv' >> res1.csv
echo $qwen  | jq -r '[.model, "default", .message.content] | @csv' >> res1.csv

echo ''
jq -r -n --arg v1 "$llama" --arg v2 "$gemma" --arg v3 "$qwen" '([$v1, $v2, $v3].[] | fromjson | ["task1", .model, "default", .total_duration, .load_duration, .prompt_eval_count, .prompt_eval_duration, .eval_count, .eval_duration]) | @csv' >> stats.csv

### Finetuned Params
llama=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
    "stream": false,
    "options": {"temperature": 0.2, "top_k":100}
    }' localhost:11434/api/chat)

gemma=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "gemma3:270m",
    "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
    "stream": false,
    "options": {"temperature": 0.2, "top_k":100}
    }' localhost:11434/api/chat)

qwen=$(curl -X POST -H 'Content-Type: application/json' \
    -d '{
    "model": "qwen2.5:0.5b",
    "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
    "stream": false,
    "options": {"temperature": 0.2, "top_k":100}
    }' localhost:11434/api/chat)

echo $llama | jq -r '[.model, "finetuned", .message.content] | @csv' >> res1.csv
echo $gemma | jq -r '[.model, "finetuned", .message.content] | @csv' >> res1.csv
echo $qwen  | jq -r '[.model, "finetuned", .message.content] | @csv' >> res1.csv

jq -r -n --arg v1 "$llama" --arg v2 "$gemma" --arg v3 "$qwen" '([$v1, $v2, $v3].[] | fromjson | ["task1", .model, "finetuned", .total_duration, .load_duration, .prompt_eval_count, .prompt_eval_duration, .eval_count, .eval_duration]) | @csv' >> stats.csv


