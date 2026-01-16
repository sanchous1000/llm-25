#!/bin/bash

prompt=$1

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-no-key-required" \
  -d '{
    "model": "local-llama",
    "messages": [
      { "role": "user", "content": "'"$prompt"'" }
    ]
  }'