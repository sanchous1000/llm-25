PROMPTS=(
    "Washington Capitals has won its second Stanley Cup this season!"
    "Donald Trump was shot in the ear during the meeting with people in Georgia!"
    "Tsunami has hit the eastern coast of Japan. 30 people injured, expecting a higher waves in Russian Far East."
)

SYS_PROMPT="You are an experienced news classifier. Return the label from the list, which fits best to the provided example. The label list: SPORT, POLITICS, EMERGENCY. Return label only, nothing else."

for PROMPT in "${PROMPTS[@]}"; do
    ### DEFAULT PARAMS
    llama=$(curl -X POST -H 'Content-Type: application/json' \
        -d '{
            "model": "llama3.2:1b",
            "messages": [
                {"role": "system", "content": "'"$SYS_PROMPT"'"},
                {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "stream": false
            }' localhost:11434/api/chat)

    gemma=$(curl -X POST -H 'Content-Type: application/json' \
        -d '{
            "model": "gemma3:270m",
            "messages": [
                {"role": "system", "content": "'"$SYS_PROMPT"'"},
                {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "stream": false
            }' localhost:11434/api/chat)

    qwen=$(curl -X POST -H 'Content-Type: application/json' \
        -d '{
            "model": "qwen2.5:0.5b",
            "messages": [
                {"role": "system", "content": "'"$SYS_PROMPT"'"},
                {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "stream": false
            }' localhost:11434/api/chat)

    echo $llama | jq -r --arg p "$PROMPT" '[.model, "default", $p, .message.content] | @csv' >> res2.csv
    echo $gemma | jq -r --arg p "$PROMPT" '[.model, "default", $p, .message.content] | @csv' >> res2.csv
    echo $qwen  | jq -r --arg p "$PROMPT" '[.model, "default", $p, .message.content] | @csv' >> res2.csv

    echo ''
    jq -r -n --arg v1 "$llama" --arg v2 "$gemma" --arg v3 "$qwen" '([$v1, $v2, $v3].[] | fromjson | ["task2", .model, "default", .total_duration, .load_duration, .prompt_eval_count, .prompt_eval_duration, .eval_count, .eval_duration]) |  @csv' >> stats.csv

    ### FINETUNED PARAMS
    llama=$(curl -X POST -H 'Content-Type: application/json' \
        -d '{
            "model": "llama3.2:1b",
            "messages": [
                {"role": "system", "content": "'"$SYS_PROMPT"'"},
                {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "stream": false,
            "options": {"temperature": 0.2, "top_k":100}
            }' localhost:11434/api/chat)

    gemma=$(curl -X POST -H 'Content-Type: application/json' \
        -d '{
            "model": "gemma3:270m",
            "messages": [
                {"role": "system", "content": "'"$SYS_PROMPT"'"},
                {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "stream": false,
            "options": {"temperature": 0.2, "top_k":100}
            }' localhost:11434/api/chat)

    qwen=$(curl -X POST -H 'Content-Type: application/json' \
        -d '{
            "model": "qwen2.5:0.5b",
            "messages": [
                {"role": "system", "content": "'"$SYS_PROMPT"'"},
                {"role": "user", "content": "'"$PROMPT"'"}
            ],
            "stream": false,
            "options": {"temperature": 0.2, "top_k":100}
            }' localhost:11434/api/chat)

    # echo "task, model, prompt, result" >> res2.csv
    echo $llama | jq -r --arg p "$PROMPT" '[.model, "finetuned", $p, .message.content] | @csv' >> res2.csv
    echo $gemma | jq -r --arg p "$PROMPT" '[.model, "finetuned", $p, .message.content] | @csv' >> res2.csv
    echo $qwen  | jq -r --arg p "$PROMPT" '[.model, "finetuned", $p, .message.content] | @csv' >> res2.csv

    echo ''
    jq -r -n --arg v1 "$llama" --arg v2 "$gemma" --arg v3 "$qwen" '([$v1, $v2, $v3].[] | fromjson | ["task2", .model, "finetuned", .total_duration, .load_duration, .prompt_eval_count, .prompt_eval_duration, .eval_count, .eval_duration]) |  @csv' >> stats.csv
done
