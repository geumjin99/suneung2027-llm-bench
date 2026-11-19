#!/usr/bin/env bash
# Download candidate models into the local Hugging Face cache.
set -u
PY=${PYTHON:-python}
MODELS=(
  naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B
  Qwen/Qwen3.5-0.8B
  LGAI-EXAONE/EXAONE-4.0-1.2B
  Qwen/Qwen3.5-2B
  google/gemma-4-E2B-it
  kakaocorp/kanana-2-3b-instruct
  Qwen/Qwen3.5-4B
  google/gemma-4-E4B-it
  kakaocorp/kanana-1.5-8b-instruct-2505
  Qwen/Qwen3.5-9B
  google/gemma-4-12B-it
  Qwen/Qwen3-14B
)
for m in "${MODELS[@]}"; do
  echo "=== $m $(date +%T)"
  $PY -c "from huggingface_hub import snapshot_download as s; p=s('$m', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','*.py','tokenizer*']); print(p)" || echo "FAILED $m"
done
echo ALL_DONE
