#!/usr/bin/env bash
# 效率画像快速批处理：在多个数据集上用「小采样」跑 PRV，只求三项效率指标 + 粗略 EM，耗时远小于全量。
#
# 用法：
#   bash scripts/run_prv_efficiency_profile_batch.sh
#
# 环境变量：
#   DATA_DIR          数据目录（默认同 run_prv_figure4_batch.sh）
#   SAMPLE_SIZE       每数据集采样条数（默认 100，可 50/200）
#   PARALLEL_JOBS     并行数，>1 时多数据集同时跑（默认 2，视 vLLM 负载调整）
#   DATASETS_PROFILE  空格分隔的 jsonl 文件名，默认 8 个常用数据集
#   RECOVERY_MODE     origin（默认）或 improved，与 run_prv_eval --recovery-mode 一致；输出文件名会带该后缀便于对比
#
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
EXP_DIR="${EXP_DIR:-$BASE/../experiments/PRV}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/results}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
EVAL_TOP_K="${EVAL_TOP_K:-40}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
RECOVERY_MODE="${RECOVERY_MODE:-origin}"

if [ -n "${DATASETS_PROFILE:-}" ]; then
  read -ra DATASETS <<< "$DATASETS_PROFILE"
else
  DATASETS=(
    "nq.jsonl"
    "2wikimqa.jsonl"
    "hotpotqa.jsonl"
    "eli5.jsonl"
    "fever.jsonl"
    "arc_challenge.jsonl"
    "asqa_eval_gtr_top100.jsonl"
    "popqa.jsonl"
  )
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M)"
RUN_ID="prv_efficiency_profile_${RECOVERY_MODE}_${ts}"
RUN_DIR="${EXP_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}/results" "${RUN_DIR}/logs"

echo "Efficiency profile batch: recovery_mode=$RECOVERY_MODE, sample_size=$SAMPLE_SIZE, parallel=$PARALLEL_JOBS, datasets=${#DATASETS[@]}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
if ! curl -sf --connect-timeout 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "Error: vLLM not reachable at http://${VLLM_HOST}:${VLLM_PORT}"
  exit 1
fi

PIDS=()
for f in "${DATASETS[@]}"; do
  IN="${DATA_DIR}/${f}"
  base="${f%.jsonl}"
  OUT_JSON="${OUT_DIR}/prv_profile_${base}_n${SAMPLE_SIZE}_${RECOVERY_MODE}.json"
  LOG_FILE="${LOG_DIR}/prv_profile_${base}_n${SAMPLE_SIZE}_${RECOVERY_MODE}_${ts}.log"

  if [ ! -f "$IN" ]; then
    echo "[WARN] skip (missing): $IN"
    continue
  fi

  if [ -s "$OUT_JSON" ]; then
    echo "[SKIP] exists: $OUT_JSON"
  else
    echo "[RUN] $base (n=$SAMPLE_SIZE, recovery=$RECOVERY_MODE)" | tee -a "$LOG_FILE"
    if [ "${PARALLEL_JOBS}" -gt 1 ]; then
      python "$BASE/evaluation/run_prv_eval.py" \
        --input-jsonl "$IN" \
        --output-json "$OUT_JSON" \
        --eval-top-k "$EVAL_TOP_K" \
        --sample-size "$SAMPLE_SIZE" \
        --shuffle \
        --seed 42 \
        --recovery-mode "$RECOVERY_MODE" \
        >> "$LOG_FILE" 2>&1 &
      PIDS+=($!)
    else
      python "$BASE/evaluation/run_prv_eval.py" \
        --input-jsonl "$IN" \
        --output-json "$OUT_JSON" \
        --eval-top-k "$EVAL_TOP_K" \
        --sample-size "$SAMPLE_SIZE" \
        --shuffle \
        --seed 42 \
        --recovery-mode "$RECOVERY_MODE" \
        >> "$LOG_FILE" 2>&1
    fi
  fi
done

if [ "${#PIDS[@]}" -gt 0 ]; then
  echo "[WAIT] ${#PIDS[@]} parallel job(s)..."
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
fi

# 收集效率画像路径并打印汇总
echo ""
echo "--- Efficiency profiles (per-dataset, recovery_mode=$RECOVERY_MODE) ---"
for f in "${DATASETS[@]}"; do
  base="${f%.jsonl}"
  OUT_JSON="${OUT_DIR}/prv_profile_${base}_n${SAMPLE_SIZE}_${RECOVERY_MODE}.json"
  PROFILE_JSON="${OUT_DIR}/prv_profile_${base}_n${SAMPLE_SIZE}_${RECOVERY_MODE}_efficiency_profile.json"
  if [ -s "$PROFILE_JSON" ]; then
    echo "  $base: $PROFILE_JSON"
    cp -f "$PROFILE_JSON" "${RUN_DIR}/results/" 2>/dev/null || true
  fi
  [ -s "$OUT_JSON" ] && cp -f "$OUT_JSON" "${RUN_DIR}/results/" 2>/dev/null || true
done
echo "[DONE] Efficiency profile batch archived to ${RUN_DIR}"
