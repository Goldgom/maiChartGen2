#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-datasets}"
CACHE_ROOT="${CACHE_ROOT:-/data/maiG_v2/cache}"
CONFIG="${CONFIG:-configs/rotating_4090.yaml}"
RUN_DIR="${RUN_DIR:-/data/maiG_v2/runs/rotating_4090}"
PYTHON="${PYTHON:-python}"
LOG_FILE="${LOG_FILE:-terminal.log}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAXSUBDIV="${MAXSUBDIV:-64}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
ENCODEC_LAYERS="${ENCODEC_LAYERS:-1}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
REFINE_EPOCHS="${REFINE_EPOCHS:-1}"
LIMIT_ARGS=()
export DATA_ROOT CACHE_ROOT CONFIG RUN_DIR

LOG_DIR="$(dirname "$LOG_FILE")"
if [[ "$LOG_DIR" != "." ]]; then
  mkdir -p "$LOG_DIR"
fi
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; echo "train_from_zero finished: $(date -Is) exit_status=$status"' EXIT

echo "============================================================"
echo "train_from_zero started: $(date -Is)"
echo "log_file=$LOG_FILE"
echo "data_root=$DATA_ROOT"
echo "cache_root=$CACHE_ROOT"
echo "config=$CONFIG"
echo "run_dir=$RUN_DIR"
echo "python=$PYTHON"
echo "num_workers=$NUM_WORKERS"
echo "maxsubdiv=$MAXSUBDIV"
echo "max_tokens=$MAX_TOKENS"
echo "stage1_epochs=$STAGE1_EPOCHS"
echo "refine_epochs=$REFINE_EPOCHS"
echo "============================================================"

if [[ "${1:-}" == "--smoke-limit" ]]; then
  LIMIT_ARGS=(--limit "${2:?missing limit value}")
fi

echo "[1/7] Preprocess audio + labels"
"$PYTHON" scripts/preprocess_all.py \
  --steps audio,labels \
  --data-root "$DATA_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --num-workers "$NUM_WORKERS" \
  --maxsubdiv "$MAXSUBDIV" \
  --max-tokens "$MAX_TOKENS" \
  --encodec-layers "$ENCODEC_LAYERS" \
  --force \
  "${LIMIT_ARGS[@]}"

echo "[2/7] Train stage1"
"$PYTHON" train.py \
  --config "$CONFIG" \
  --train-stage stage1 \
  --max-epochs "$STAGE1_EPOCHS"

STAGE1_CKPT="${STAGE1_CKPT:-$RUN_DIR/stage1/best.pt}"
if [[ ! -f "$STAGE1_CKPT" ]]; then
  STAGE1_CKPT="$RUN_DIR/stage1/last.pt"
fi

echo "[3/7] Export stage1 hidden + build downstream caches"
"$PYTHON" scripts/build_stage234_cache.py \
  --step all \
  --checkpoint "$STAGE1_CKPT" \
  --config "$CONFIG" \
  --cache-root "$CACHE_ROOT" \
  "${LIMIT_ARGS[@]}"

echo "[4/7] Build event-stage caches"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
import yaml
from scripts.build_stage234_cache import _build_event_stage_caches

cfg_path = Path(os.environ.get("CONFIG", "configs/rotating_4090.yaml"))
cache_root = Path(os.environ.get("CACHE_ROOT", "/data/maiG_v2/cache"))
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
count = _build_event_stage_caches(cache_root, cfg)
print(f"event-stage caches: {count}")
PY

echo "[5/7] Train refinement stages"
for STAGE in stage2_star hold touch_hold stage5_touch stage6_break_note stage7_firework_note; do
  echo "  -> $STAGE"
  "$PYTHON" train.py \
    --config "$CONFIG" \
    --train-stage "$STAGE" \
    --max-epochs "$REFINE_EPOCHS"
done

echo "[6/7] Optional legacy stages"
if [[ "${TRAIN_LEGACY_STAGES:-0}" == "1" ]]; then
  for STAGE in touch slide break spike; do
    echo "  -> $STAGE"
    "$PYTHON" train.py \
      --config "$CONFIG" \
      --train-stage "$STAGE" \
      --max-epochs "$REFINE_EPOCHS"
  done
fi

echo "[7/7] Done"
