#!/usr/bin/env bash
# Run a list of models sequentially on one GPU, waiting for each model's download to finish.
# Usage: bash run_queue.sh <items.json[,items2.json]> <model> [<model> ...]
ITEMS=$1; shift
HUB=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
ready() {  # snapshot exists, no partial blobs, every shard named in the index is present
  local d=$HUB/models--${1//\//--}
  [ -d "$d/snapshots" ] || return 1
  ls "$d/blobs" 2>/dev/null | grep -q incomplete && return 1
  python3 - "$d" <<'PY'
import json, glob, os, sys
d = sys.argv[1]; snap = sorted(glob.glob(d + "/snapshots/*"))[-1]
idx = os.path.join(snap, "model.safetensors.index.json")
need = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
sys.exit(0 if all(os.path.exists(os.path.join(snap, f)) for f in need) else 1)
PY
}
for m in "$@"; do
  until ready "$m"; do sleep 30; done
  echo "=== $m $(date +%T)"
  python run_eval.py "$m" "$ITEMS" ${OUT:-../out/raw} --n 3 || echo "FAILED $m"
  # DELETE_AFTER="repo1 repo2": free disk once every item file has a result for this model
  if [[ " ${DELETE_AFTER:-} " == *" $m "* ]]; then
    ok=1; for f in ${ITEMS//,/ }; do [ -f "${OUT:-../out/raw}/${m//\//__}__$(basename "$f")" ] || ok=0; done
    [ $ok = 1 ] && rm -rf "$HUB/models--${m//\//--}" && echo "deleted cache $m"
  fi
done
echo QUEUE_DONE
