#!/usr/bin/env bash
# 先跑几条测一下：单数据集、少量样本，确认环境和输出再跑全量。
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
DYNAMICRAG_DIR="${DYNAMICRAG_DIR:-$BASE/../DynamicRAG}"
OUT_DIR="${OUT_DIR:-$BASE/results}"
LOG_DIR="${LOG_DIR:-$BASE/logs}"
SAMPLE_SIZE="${SAMPLE_SIZE:-5}"
DATASET="${DATASET:-nq.jsonl}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# 检查 vLLM 是否已启动（PRV 依赖 REAP-all-lora 的 vLLM 服务）
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}"
if ! curl -sf --connect-timeout 3 "${VLLM_URL}/v1/models" >/dev/null 2>&1; then
  echo "Error: vLLM is not reachable at ${VLLM_URL}"
  echo "Please start the model server first, e.g.:"
  echo "  vllm serve /path/to/REAP-all-merged --host 0.0.0.0 --port ${VLLM_PORT}"
  echo "  export VLLM_LLM_MODEL=/path/to/REAP-all-merged  # 与 vLLM 加载的模型一致"
  echo "Then run this script again."
  exit 1
fi
echo "vLLM OK: ${VLLM_URL}"

IN="${DATA_DIR}/${DATASET}"
base="${DATASET%.jsonl}"
OUT_JSON="${OUT_DIR}/prv_quick_test_${base}.json"
LOG_FILE="${LOG_DIR}/prv_quick_test_${base}.log"

if [ ! -f "$IN" ]; then
  echo "Error: input not found: $IN (set DATA_DIR if needed)"
  exit 1
fi

echo "Running PRV quick test: dataset=$DATASET sample_size=$SAMPLE_SIZE (full PRV: planning + E5-on-ctxs)"
python "$BASE/evaluation/run_prv_eval.py" \
  --input-jsonl "$IN" \
  --output-json "$OUT_JSON" \
  --eval-top-k 40 \
  --sample-size "$SAMPLE_SIZE" \
  2>&1 | tee "$LOG_FILE"

if [ -s "$OUT_JSON" ] && [ -f "${DYNAMICRAG_DIR}/evaluate.py" ]; then
  echo "--- EM on $SAMPLE_SIZE samples ---"
  python "${DYNAMICRAG_DIR}/evaluate.py" --results_file "$OUT_JSON" --metric em
else
  echo "Output: $OUT_JSON (evaluate skipped if DynamicRAG not found)"
fi
