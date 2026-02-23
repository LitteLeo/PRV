#!/usr/bin/env bash
# PRV 消融实验：依次跑 Full PRV、No Planning、REAP 基线，结果存到 /home/lfy/experiments/PRV/8b_v2_batch_<timestamp>/results
# - Full PRV：REAP + 重排 + 统一模型（规划→重排→校验）
# - No Planning：单轮 重排→事实提取→合成
# - REAP 基线：REAP + 统一模型、无重排（规划→检索 top-k→事实提取→合成），用于对比「重排」的贡献
# 全部串行执行，每个数据集 50 条，seed 42。
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
EXP_BASE="${EXP_BASE:-/home/lfy/experiments/PRV}"
ts="$(date +%Y%m%d_%H%M)"
RUN_DIR="${EXP_BASE}/8b_v2_batch_${ts}"
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

DATASETS=(
  "nq.jsonl"
  "2wikimqa.jsonl"
  "hotpotqa.jsonl"
  "eli5.jsonl"
  "fever.jsonl"
  "arc_challenge.jsonl"
  "asqa_eval_gtr_top100.jsonl"
  "popqa.jsonl"
  "triviaqa.jsonl"
)

EVAL_ARGS="--eval-top-k 40 --sample-size 50 --shuffle --seed 42"

for f in "${DATASETS[@]}"; do
  IN="${DATA_DIR}/${f}"
  base="${f%.jsonl}"
  if [ ! -f "$IN" ]; then
    echo "[WARN] missing $IN, skip."
    continue
  fi

  echo "[RUN] Full PRV: $base"
  OUT_FULL="${OUT_DIR}/prv_8b_v2_${base}_50.json"
  python "$BASE/evaluation/run_prv_eval.py" \
    --input-jsonl "$IN" \
    --output-json "$OUT_FULL" \
    $EVAL_ARGS 2>&1 | tee -a "${LOG_DIR}/prv_8b_v2_${base}_50_full.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

  echo "[RUN] No Planning: $base"
  OUT_NOP="${OUT_DIR}/prv_8b_v2_${base}_50_no_planning.json"
  python "$BASE/evaluation/run_prv_eval.py" --no-planning \
    --input-jsonl "$IN" \
    --output-json "$OUT_NOP" \
    $EVAL_ARGS 2>&1 | tee -a "${LOG_DIR}/prv_8b_v2_${base}_50_no_planning.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

  echo "[RUN] REAP 基线 (无重排): $base"
  OUT_REAP_BASELINE="${OUT_DIR}/prv_8b_v2_${base}_50_reap_baseline.json"
  USE_PRV_RERANK=false python "$BASE/evaluation/run_prv_eval.py" \
    --input-jsonl "$IN" \
    --output-json "$OUT_REAP_BASELINE" \
    $EVAL_ARGS 2>&1 | tee -a "${LOG_DIR}/prv_8b_v2_${base}_50_reap_baseline.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
done

echo "[DONE] Ablation results in $OUT_DIR"