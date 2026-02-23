#!/usr/bin/env bash
# 并行跑多个数据集的 PRV 评测；Ctrl+C 会结束所有子进程。
# 用法：VLLM_LLM_MODEL=/home/lfy/projects/models/REAP-all-merged bash scripts/run_prv_v2_parallel.sh
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
# 实验根目录：默认将并行评测结果写到代码仓库外的 experiments/PRV 下
EXP_DIR="${EXP_DIR:-$BASE/../experiments/PRV}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/results}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"
# 并行数，可按 GPU 能力调整
PARALLEL="${PARALLEL:-2}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

# 子进程 PIDs
PIDS=()
cleanup() {
  echo ""
  echo "Stopping all eval processes..."
  for pid in "${PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait "${PIDS[@]}" 2>/dev/null || true
  exit 130
}
trap cleanup SIGINT SIGTERM

# 检查 vLLM
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
if ! curl -sf --connect-timeout 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "Error: vLLM not reachable. Start vLLM and set VLLM_LLM_MODEL to match the loaded model."
  exit 1
fi
echo "VLLM_LLM_MODEL=${VLLM_LLM_MODEL:-<not set, using config default>}"
echo "Running up to $PARALLEL datasets in parallel. Ctrl+C to stop all."

# 数据集列表：若已设置环境变量 DATASETS（空格分隔），则用其；否则用默认全量列表
if [ -n "${DATASETS:-}" ]; then
  read -ra DATASETS <<< "$DATASETS"
else
  DATASETS=(nq.jsonl 2wikimqa.jsonl hotpotqa.jsonl eli5.jsonl fever.jsonl arc_challenge.jsonl asqa_eval_gtr_top100.jsonl popqa.jsonl triviaqa.jsonl)
fi

ts="$(date +%Y%m%d_%H%M)"

for f in "${DATASETS[@]}"; do
  IN="${DATA_DIR}/${f}"
  base="${f%.jsonl}"
  OUT_JSON="${OUT_DIR}/prv_8b_v2_${base}.json"
  LOG_FILE="${LOG_DIR}/prv_eval_8b_v2_${base}_${ts}.log"
  if [ ! -f "$IN" ]; then
    echo "[WARN] skip (no file): $IN"
    continue
  fi
  if [ -s "$OUT_JSON" ]; then
    echo "[SKIP] exists: $OUT_JSON"
    continue
  fi
  echo "[RUN] $base -> $OUT_JSON"
  python "$BASE/evaluation/run_prv_eval.py" \
    --input-jsonl "$IN" \
    --output-json "$OUT_JSON" \
    --eval-top-k 40 \
    >> "$LOG_FILE" 2>&1 &
  PIDS+=($!)
  # 控制并行数：若已有 PARALLEL 个在跑，先等一个完成
  while [ ${#PIDS[@]} -ge "$PARALLEL" ]; do
    for i in "${!PIDS[@]}"; do
      if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
        wait "${PIDS[i]}" 2>/dev/null || true
        unset 'PIDS[i]'
        PIDS=("${PIDS[@]}")
        break 2
      fi
    done
    sleep 2
  done
done

echo "Waiting for remaining jobs..."
wait "${PIDS[@]}"
echo "Done."
