#!/usr/bin/env bash
# 四检索器 × 三 benchmark 一次性跑 PRV：DPR、Contriever、MonoT5、E5 × NQ/HotpotQA/ASQA，抽样 50，seed 42。
#
# 用法一（先检索再跑 PRV）：
#   bash scripts/retrieve_four_retrievers.sh   # 生成 12 个 JSONL 到 DATA_DIR
#   bash scripts/run_prv_four_retrievers_batch.sh
#
# 用法二（已有 12 个 JSONL）：
#   export DATA_DIR=/path/to/eval_data
#   bash scripts/run_prv_four_retrievers_batch.sh
#
# 试跑一条（只跑 1 个数据集、2 条样本，用于检查是否报错）：
#   TEST=1 bash scripts/run_prv_four_retrievers_batch.sh
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
DYNAMICRAG_DIR="${DYNAMICRAG_DIR:-$(cd "$BASE/../DynamicRAG" 2>/dev/null && pwd)}"
EXP_DIR="${EXP_DIR:-$BASE/../experiments/PRV}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/results}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"

# 试跑：只跑 1 个数据集、2 条样本
if [ "${TEST:-0}" = "1" ]; then
  SAMPLE_SIZE="${SAMPLE_SIZE:-2}"
  # 优先用已有 nq.jsonl，没有则用 nq_contriever.jsonl
  DATASETS=("nq.jsonl")
  echo "[TEST] 试跑 1 个数据集、${SAMPLE_SIZE} 条样本"
else
  SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
  EVAL_TOP_K="${EVAL_TOP_K:-40}"
  SEED=42
  # 四检索器 × 三 benchmark = 12 个数据集
  BENCHMARKS=(nq hotpotqa asqa)
  RETRIEVERS=(dpr contriever monot5 e5)
  DATASETS=()
  for b in "${BENCHMARKS[@]}"; do
    for r in "${RETRIEVERS[@]}"; do
      if [ "$b" = "asqa" ]; then
        DATASETS+=("asqa_eval_gtr_top100_${r}.jsonl")
      else
        DATASETS+=("${b}_${r}.jsonl")
      fi
    done
  done
fi

EVAL_TOP_K="${EVAL_TOP_K:-40}"
SEED="${SEED:-42}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$EXP_DIR"

# 检查 vLLM
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
if ! curl -sf --connect-timeout 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "Error: vLLM not reachable at http://${VLLM_HOST}:${VLLM_PORT}"
  exit 1
fi
echo "vLLM OK"

ts="$(date +%Y%m%d_%H%M)"
RUN_ID="prv_four_retrievers_${ts}"
RUN_EXP_DIR="${EXP_DIR}/${RUN_ID}"
mkdir -p "${RUN_EXP_DIR}/results" "${RUN_EXP_DIR}/logs"

for f in "${DATASETS[@]}"; do
  IN="${DATA_DIR}/${f}"
  base="${f%.jsonl}"
  OUT_JSON="${OUT_DIR}/prv_figure4_${base}_${SAMPLE_SIZE}.json"
  LOG_FILE="${LOG_DIR}/prv_figure4_${base}_${ts}.log"
  [ "${TEST:-0}" = "1" ] && LOG_FILE="${LOG_DIR}/prv_figure4_${base}_test.log"

  if [ ! -f "$IN" ]; then
    echo "[WARN] missing: $IN, skip." | tee -a "$LOG_FILE"
    continue
  fi

  if [ -s "$OUT_JSON" ]; then
    echo "[SKIP] exists: $OUT_JSON" | tee -a "$LOG_FILE"
  else
    echo "[RUN] PRV $base (top-k=$EVAL_TOP_K, sample=$SAMPLE_SIZE, seed=$SEED)" | tee -a "$LOG_FILE"
    python "$BASE/evaluation/run_prv_eval.py" \
      --input-jsonl "$IN" \
      --output-json "$OUT_JSON" \
      --eval-top-k "$EVAL_TOP_K" \
      --sample-size "$SAMPLE_SIZE" \
      --shuffle \
      --seed "$SEED" \
      >> "$LOG_FILE" 2>&1
  fi

  MET_FILE="${OUT_DIR}/metrics_em_prv_figure4_${base}_${SAMPLE_SIZE}.txt"
  if [ -s "$OUT_JSON" ] && [ -n "${DYNAMICRAG_DIR}" ] && [ -f "${DYNAMICRAG_DIR}/evaluate.py" ]; then
    python "${DYNAMICRAG_DIR}/evaluate.py" --results_file "$OUT_JSON" --metric em | tee "$MET_FILE"
  fi
  cp -f "$OUT_JSON" "${RUN_EXP_DIR}/results/" 2>/dev/null || true
  [ -f "$MET_FILE" ] && cp -f "$MET_FILE" "${RUN_EXP_DIR}/results/" || true
  [ -f "$LOG_FILE" ] && cp -f "$LOG_FILE" "${RUN_EXP_DIR}/logs/" || true
done

echo "[DONE] PRV four retrievers archived to ${RUN_EXP_DIR}"
