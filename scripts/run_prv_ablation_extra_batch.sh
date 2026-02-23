#!/usr/bin/env bash
# PRV 补消融实验（排除已跑完的 Full PRV / No Planning / No Rerank 以及 Rerank-only）：
#   1. w/o RePlanner (no recovery)：PARTIAL/FAILED 时不修复计划，直接终止并合成
#   2. w/o Verification constraints：FE 不输出结构化证据/满足度标签，仅自由文本结论
#   3. Planning-only loop：保留计划更新与多轮检索，不做结构化事实记忆（REAP-like）
# 结果存到 ${EXP_BASE}/8b_v2_ablation_extra_<timestamp>/results，每数据集 50 条，seed 42。
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
EXP_BASE="${EXP_BASE:-/home/lfy/experiments/PRV}"
ts="$(date +%Y%m%d_%H%M)"
RUN_DIR="${EXP_BASE}/8b_v2_ablation_extra_${ts}"
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

  echo "[RUN] w/o RePlanner (no recovery): $base"
  OUT_NORECOV="${OUT_DIR}/prv_8b_v2_${base}_50_no_recovery.json"
  python "$BASE/evaluation/run_prv_eval.py" --no-recovery \
    --input-jsonl "$IN" \
    --output-json "$OUT_NORECOV" \
    $EVAL_ARGS 2>&1 | tee -a "${LOG_DIR}/prv_8b_v2_${base}_50_no_recovery.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

  echo "[RUN] w/o Verification constraints: $base"
  OUT_NOVERIF="${OUT_DIR}/prv_8b_v2_${base}_50_no_verification.json"
  USE_VERIFICATION_CONSTRAINTS=false python "$BASE/evaluation/run_prv_eval.py" \
    --input-jsonl "$IN" \
    --output-json "$OUT_NOVERIF" \
    $EVAL_ARGS 2>&1 | tee -a "${LOG_DIR}/prv_8b_v2_${base}_50_no_verification.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1

  echo "[RUN] Planning-only loop: $base"
  OUT_PLANONLY="${OUT_DIR}/prv_8b_v2_${base}_50_planning_only.json"
  python "$BASE/evaluation/run_prv_eval.py" --planning-only \
    --input-jsonl "$IN" \
    --output-json "$OUT_PLANONLY" \
    $EVAL_ARGS 2>&1 | tee -a "${LOG_DIR}/prv_8b_v2_${base}_50_planning_only.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
done

echo "[DONE] Extra ablation results in $OUT_DIR"
