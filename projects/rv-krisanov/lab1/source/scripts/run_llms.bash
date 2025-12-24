#!/bin/bash

ollama run mistral:7b
OLLAMA_HOST=0.0.0.0:11434 ollama run qwen3:4b