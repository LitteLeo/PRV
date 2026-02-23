#!/usr/bin/env bash
# PRV Figure4 实验：在 NQ、HotpotQA、ASQA 上跑 PRV 评测并出 EM。
# 与 run_prv_v2_batch.sh 逻辑一致，默认只跑这三个 benchmark；支持多检索器多文件。
# 无需改 Python 代码。
#
# 运行前准备：
#   1. 启动 vLLM：vllm serve /path/to/REAP-all-merged --host 0.0.0.0 --port 8000
#   2. （可选）E5 编码器路径：export E5_ENCODER_PATH=/path/to/e5-large-v2  # 默认见 config.py
#   3. 数据目录：export DATA_DIR=/path/to/eval_data  # 需含 nq.jsonl, hotpotqa.jsonl, asqa_eval_gtr_top100.jsonl
#
# 运行（在 PRV 目录下）：
#   bash scripts/run_prv_figure4_batch.sh
#
# 四检索器对比（需先准备好 12 个 JSONL）：
#   export DATA_DIR=/path/to/eval_data
#   export DATASETS_FIGURE4="nq_dpr.jsonl nq_contriever.jsonl nq_e5.jsonl nq_monot5.jsonl hotpotqa_dpr.jsonl hotpotqa_contriever.jsonl hotpotqa_e5.jsonl hotpotqa_monot5.jsonl asqa_dpr.jsonl asqa_contriever.jsonl asqa_e5.jsonl asqa_monot5.jsonl"
#   bash scripts/run_prv_figure4_batch.sh
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
DYNAMICRAG_DIR="${DYNAMICRAG_DIR:-$(cd "$BASE/../DynamicRAG" 2>/dev/null && pwd)}"
EXP_DIR="${EXP_DIR:-$BASE/../experiments/PRV}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/results}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"

# 采样与 top-k：采样 50 条、shuffle、seed 固定 42；eval-top-k 可覆盖
# 全量跑设 SAMPLE_SIZE= 或 0
# 多数据集并行：PARALLEL_JOBS>1 可缩短总时间；80GB 显存建议≤2，4 路易 OOM
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
EVAL_TOP_K="${EVAL_TOP_K:-40}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"

# Figure4 三个 benchmark。四检索器时用环境变量传 12 个文件：
#   export DATASETS_FIGURE4="nq_dpr.jsonl nq_contriever.jsonl nq_e5.jsonl nq_monot5.jsonl hotpotqa_dpr.jsonl hotpotqa_contriever.jsonl hotpotqa_e5.jsonl hotpotqa_monot5.jsonl asqa_dpr.jsonl asqa_contriever.jsonl asqa_e5.jsonl asqa_monot5.jsonl"
if [ -n "${DATASETS_FIGURE4:-}" ]; then
  read -ra DATASETS <<< "$DATASETS_FIGURE4"
else
  DATASETS=("nq.jsonl" "hotpotqa.jsonl" "asqa_eval_gtr_top100.jsonl")
fi

mkdir -p "$OUT_DIR" "$LOG_DIR" "$EXP_DIR"

# 检查 vLLM
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
if ! curl -sf --connect-timeout 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "Error: vLLM not reachable at http://${VLLM_HOST}:${VLLM_PORT}"
  echo "Start first: vllm serve /path/to/REAP-all-merged --host 0.0.0.0 --port ${VLLM_PORT}"
  exit 1
fi
echo "vLLM OK at http://${VLLM_HOST}:${VLLM_PORT}"

ts="$(date +%Y%m%d_%H%M)"
RUN_ID="prv_figure4_${ts}"
RUN_EXP_DIR="${EXP_DIR}/${RUN_ID}"
mkdir -p "${RUN_EXP_DIR}/results" "${RUN_EXP_DIR}/logs"

echo "PYTHON=$(which python)" > "${RUN_EXP_DIR}/logs/env.txt"
python -V >> "${RUN_EXP_DIR}/logs/env.txt" 2>&1 || true

# 第一阶段：按需启动 PRV（串行 PARALLEL_JOBS=1 或并行 PARALLEL_JOBS>1）
PIDS=()
for f in "${DATASETS[@]}"; do
  IN="${DATA_DIR}/${f}"
  base="${f%.jsonl}"
  if [ -n "${SAMPLE_SIZE:-}" ] && [ "$SAMPLE_SIZE" -gt 0 ]; then
    OUT_JSON="${OUT_DIR}/prv_figure4_${base}_${SAMPLE_SIZE}.json"
  else
    OUT_JSON="${OUT_DIR}/prv_figure4_${base}.json"
  fi
  LOG_FILE="${LOG_DIR}/prv_figure4_${base}_${ts}.log"

  if [ ! -f "$IN" ]; then
    echo "[WARN] missing: $IN, skip." | tee -a "$LOG_FILE"
    continue
  fi

  if [ -s "$OUT_JSON" ]; then
    echo "[SKIP] exists: $OUT_JSON" | tee -a "$LOG_FILE"
  else
    echo "[RUN] PRV $base (top-k=$EVAL_TOP_K, sample=$SAMPLE_SIZE, seed=42)" | tee -a "$LOG_FILE"
    if [ "${PARALLEL_JOBS}" -gt 1 ]; then
      python "$BASE/evaluation/run_prv_eval.py" \
        --input-jsonl "$IN" \
        --output-json "$OUT_JSON" \
        --eval-top-k "$EVAL_TOP_K" \
        --sample-size "$SAMPLE_SIZE" \
        --shuffle \
        --seed 42 \
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
        >> "$LOG_FILE" 2>&1
    fi
  fi
done
if [ "${#PIDS[@]}" -gt 0 ]; then
  echo "[WAIT] ${#PIDS[@]} parallel job(s)..."
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
fi

# 第二阶段：打 EM 并归档
for f in "${DATASETS[@]}"; do
  base="${f%.jsonl}"
  if [ -n "${SAMPLE_SIZE:-}" ] && [ "$SAMPLE_SIZE" -gt 0 ]; then
    OUT_JSON="${OUT_DIR}/prv_figure4_${base}_${SAMPLE_SIZE}.json"
    MET_FILE="${OUT_DIR}/metrics_em_prv_figure4_${base}_${SAMPLE_SIZE}.txt"
  else
    OUT_JSON="${OUT_DIR}/prv_figure4_${base}.json"
    MET_FILE="${OUT_DIR}/metrics_em_prv_figure4_${base}.txt"
  fi
  LOG_FILE="${LOG_DIR}/prv_figure4_${base}_${ts}.log"
  if [ -s "$OUT_JSON" ] && [ -n "${DYNAMICRAG_DIR}" ] && [ -f "${DYNAMICRAG_DIR}/evaluate.py" ]; then
    python "${DYNAMICRAG_DIR}/evaluate.py" --results_file "$OUT_JSON" --metric em | tee "$MET_FILE"
  elif [ -s "$OUT_JSON" ]; then
    echo "[WARN] DynamicRAG/evaluate.py not found, skip metric." | tee "$MET_FILE"
  fi
  cp -f "$OUT_JSON" "${RUN_EXP_DIR}/results/" 2>/dev/null || true
  [ -f "$MET_FILE" ] && cp -f "$MET_FILE" "${RUN_EXP_DIR}/results/" || true
  [ -f "$LOG_FILE" ] && cp -f "$LOG_FILE" "${RUN_EXP_DIR}/logs/" || true
done

echo "[DONE] PRV Figure4 batch archived to ${RUN_EXP_DIR}"
