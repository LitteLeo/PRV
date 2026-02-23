#!/usr/bin/env bash
# PRV Top-N 小 K 实验（仅 NQ）：zeroshot + top5 / top10 / top20 / top30 / top40。
# 与 run_prv_topn_batch.sh 互补：后者跑 top50/100/150/200/300/500 + NQ/Hotpot/ASQA；
# 本脚本只跑 NQ，Top-N = 5,10,20,30,40，抽样 50 条、seed 42、跑 1 轮。
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
EXP_BASE="${EXP_BASE:-/home/lfy/experiments/PRV}"
ts="$(date +%Y%m%d_%H%M)"
RUN_DIR="${EXP_BASE}/topn_nq_lowk_8b_v2_${ts}"
OUT_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "OUT_DIR=$OUT_DIR"
echo "LOG_DIR=$LOG_DIR"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
if ! curl -sf --connect-timeout 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "Warning: vLLM not reachable at http://${VLLM_HOST}:${VLLM_PORT}. Continue anyway."
else
  echo "vLLM OK at http://${VLLM_HOST}:${VLLM_PORT}"
fi

# 仅 NQ
DATASETS=( "nq.jsonl" )
TOP_N_LIST=( 5 10 20 30 40 )
NUM_RUNS=1
SAMPLE_ARGS="--sample-size 50 --shuffle --seed 42"

for f in "${DATASETS[@]}"; do
  IN="${DATA_DIR}/${f}"
  base="${f%.jsonl}"
  if [ ! -f "$IN" ]; then
    echo "[WARN] missing $IN, skip."
    continue
  fi

  # --- Zero-Shot（无检索），1 轮
  for run in $(seq 1 "$NUM_RUNS"); do
    OUT="${OUT_DIR}/prv_8b_v2_${base}_zeroshot_run${run}.json"
    LOG="${LOG_DIR}/prv_8b_v2_${base}_zeroshot_run${run}.log"
    echo "[RUN] $base zeroshot run$run"
    python "$BASE/evaluation/run_prv_eval.py" \
      --input-jsonl "$IN" \
      --output-json "$OUT" \
      --zero-shot \
      $SAMPLE_ARGS \
      2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
  done

  # --- Top-N：5, 10, 20, 30, 40，各 1 轮（抽样 50 条）
  for n in "${TOP_N_LIST[@]}"; do
    for run in $(seq 1 "$NUM_RUNS"); do
      OUT="${OUT_DIR}/prv_8b_v2_${base}_top${n}_run${run}.json"
      LOG="${LOG_DIR}/prv_8b_v2_${base}_top${n}_run${run}.log"
      echo "[RUN] $base top-$n run$run"
      python "$BASE/evaluation/run_prv_eval.py" \
        --input-jsonl "$IN" \
        --output-json "$OUT" \
        --eval-top-k "$n" \
        --max-docs "$n" \
        $SAMPLE_ARGS \
        2>&1 | tee -a "$LOG"
      [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
    done
  done
done

echo "[DONE] Top-N (low-k) NQ-only experiment results in $OUT_DIR"
