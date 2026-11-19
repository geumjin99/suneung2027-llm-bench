#!/usr/bin/env bash
# Download models into the local Hugging Face cache. Usage: bash download.sh <repo_id> [<repo_id> ...]
set -u
PY=${PYTHON:-python}
for m in "$@"; do
  echo "=== $m $(date +%T)"
  $PY -c "from huggingface_hub import snapshot_download as s; p=s('$m', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','*.py','tokenizer*']); print(p)" || echo "FAILED $m"
done
echo ALL_DONE
