#!/usr/bin/env bash
# PRV 评测：与 DynamicRAG 同一批 eval_data，用评测集自带的 ctxs。
# 默认完整 PRV：规划(拆子问题) + 多轮，每轮从 ctxs 用 E5 做相似度排序取 top-k → 重排 → 事实提取 → 更新计划 → 合成。
# 需配置 E5_ENCODER_PATH（评测进程内 E5 编码器路径）。用 --no-planning 可退化为单轮（按位置截断 ctxs）。
set -euo pipefail

# PRV 项目根目录（本脚本在 PRV/scripts/ 下）
BASE="$(cd "$(dirname "$0")/.." && pwd)"
# 评测数据目录（与 run_8b_v2_batch.sh 一致）
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
# DynamicRAG 目录（用于调用 evaluate.py）
DYNAMICRAG_DIR="${DYNAMICRAG_DIR:-$(cd "$BASE/../DynamicRAG" 2>/dev/null && pwd)}"
OUT_DIR="${OUT_DIR:-$BASE/results}"
LOG_DIR="${LOG_DIR:-$BASE/logs}"
EXP_DIR="${EXP_DIR:-$BASE/../experiments/PRV}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$EXP_DIR"

# 检查 vLLM 是否已启动
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
if ! curl -sf --connect-timeout 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "Error: vLLM not reachable at http://${VLLM_HOST}:${VLLM_PORT}"
  echo "Start the model first: vllm serve /home/lfy/projects/models/REAP-all-lora --host 0.0.0.0 --port ${VLLM_PORT}"
  exit 1
fi
echo "vLLM OK at http://${VLLM_HOST}:${VLLM_PORT}"

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

ts="$(date +%Y%m%d_%H%M)"
RUN_ID="prv_v2_batch_${ts}"
RUN_EXP_DIR="${EXP_DIR}/${RUN_ID}"
mkdir -p "${RUN_EXP_DIR}/results" "${RUN_EXP_DIR}/config" "${RUN_EXP_DIR}/logs"

echo "PYTHON=$(which python)" > "${RUN_EXP_DIR}/logs/env.txt"
python -V >> "${RUN_EXP_DIR}/logs/env.txt" 2>&1 || true
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())" >> "${RUN_EXP_DIR}/logs/env.txt" 2>&1 || true

for f in "${DATASETS[@]}"; do
  IN="${DATA_DIR}/${f}"
  base="${f%.jsonl}"
  OUT_JSON="${OUT_DIR}/prv_8b_v2_${base}.json"
  LOG_FILE="${LOG_DIR}/prv_eval_8b_v2_${base}_${ts}.log"

  if [ ! -f "$IN" ]; then
    echo "[WARN] missing input: $IN, skip." | tee -a "$LOG_FILE"
    continue
  fi

  if [ -s "$OUT_JSON" ]; then
    echo "[SKIP] exists: $OUT_JSON" | tee -a "$LOG_FILE"
  else
    echo "[RUN] PRV eval $base (planning + E5-on-ctxs)" | tee -a "$LOG_FILE"
    python "$BASE/evaluation/run_prv_eval.py" \
      --input-jsonl "$IN" \
      --output-json "$OUT_JSON" \
      --eval-top-k 40 \
      >> "$LOG_FILE" 2>&1
  fi

  MET_FILE="${OUT_DIR}/metrics_em_prv_8b_v2_${base}.txt"
  if [ -s "$OUT_JSON" ] && [ -n "${DYNAMICRAG_DIR}" ] && [ -f "${DYNAMICRAG_DIR}/evaluate.py" ]; then
    python "${DYNAMICRAG_DIR}/evaluate.py" --results_file "$OUT_JSON" --metric em | tee "$MET_FILE"
  elif [ -s "$OUT_JSON" ]; then
    echo "[WARN] DynamicRAG/evaluate.py not found at ${DYNAMICRAG_DIR:-unknown}, skip metric." | tee "$MET_FILE"
  fi

  cp -f "$OUT_JSON" "${RUN_EXP_DIR}/results/" 2>/dev/null || true
  [ -f "$MET_FILE" ] && cp -f "$MET_FILE" "${RUN_EXP_DIR}/results/" || true
  [ -f "$LOG_FILE" ] && cp -f "$LOG_FILE" "${RUN_EXP_DIR}/logs/" || true
done

echo "[DONE] PRV v2 batch eval archived to ${RUN_EXP_DIR}"
