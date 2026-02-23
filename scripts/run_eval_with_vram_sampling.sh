#!/usr/bin/env bash
# 在运行评测时同步采样 GPU 显存（nvidia-smi），得到 vLLM 服务端 Peak VRAM，用于 5.10 单卡开销对比。
#
# 用法：
#   cd PRV
#   # 先启动 vLLM（PRV 单模型 或 REAP 三模型），再在另一终端：
#   bash scripts/run_eval_with_vram_sampling.sh prv     --input-jsonl $DATA_DIR/nq.jsonl --output-json out/prv_nq.json --sample-size 50
#   bash scripts/run_eval_with_vram_sampling.sh reap    --input-jsonl $DATA_DIR/nq.jsonl --output-json out/reap_nq.json --sample-size 50
#
# 输出：除原有 JSON/profile 外，会多出 <output_json 同目录>/vram_sampling_<basename>.json，内含 server_peak_vram_mb。
# 将两次运行的 server_peak_vram_mb 填入表格即可直接体现「PRV 单卡 vs REAP 三模型」的显存优势。
#
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-}"
shift || true

if [ -z "$MODE" ] || [ "$MODE" != "prv" ] && [ "$MODE" != "reap" ]; then
  echo "Usage: $0 prv|reap  [eval args...]"
  echo "  prv:  run run_prv_eval.py with given args"
  echo "  reap: run run_reap_baseline_eval.py with given args"
  exit 1
fi

# 从参数里解析 --output-json（用于写 vram 结果）
OUT_JSON=""
for i in "$@"; do
  if [ "$i" = "--output-json" ]; then
    NEXT_IS_OUT=1
    continue
  fi
  if [ "${NEXT_IS_OUT:-0}" = "1" ]; then
    OUT_JSON="$i"
    break
  fi
done

if [ -z "$OUT_JSON" ]; then
  echo "Error: must pass --output-json <path> in eval args"
  exit 1
fi

VRAM_LOG=$(mktemp)
cleanup() { rm -f "$VRAM_LOG"; }
trap cleanup EXIT

# 默认每 1 秒采样一次；仅采样 memory.used (MB)
echo "Starting GPU memory sampling (nvidia-smi every 1s) to $VRAM_LOG ..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$VRAM_LOG" 2>/dev/null &
NVPID=$!
sleep 1

if ! kill -0 "$NVPID" 2>/dev/null; then
  echo "Warning: nvidia-smi could not start (no GPU?). Proceeding without server VRAM sampling."
  NVPID=""
fi

if [ "$MODE" = "prv" ]; then
  python "$BASE/evaluation/run_prv_eval.py" "$@"
else
  python "$BASE/evaluation/run_reap_baseline_eval.py" "$@"
fi

if [ -n "$NVPID" ]; then
  kill "$NVPID" 2>/dev/null || true
  wait "$NVPID" 2>/dev/null || true
fi

# 解析采样日志：取所有采样中的最大 memory.used (MB)
MAX_MB=""
if [ -s "$VRAM_LOG" ]; then
  MAX_MB=$(grep -oE '[0-9]+' "$VRAM_LOG" | sort -n | tail -1)
fi

OUT_DIR=$(dirname "$OUT_JSON")
BASE_NAME=$(basename "$OUT_JSON" .json)
VRAM_JSON="${OUT_DIR}/vram_sampling_${BASE_NAME}.json"

if [ -n "$MAX_MB" ] && [ "$MAX_MB" -gt 0 ]; then
  echo "{\"server_peak_vram_mb\": $MAX_MB, \"note\": \"nvidia-smi during eval (vLLM server + E5 if on same GPU)\"}" > "$VRAM_JSON"
  echo "[VRAM sampling] server_peak_vram_mb = $MAX_MB  -> $VRAM_JSON"
else
  echo "[VRAM sampling] no valid samples -> skip $VRAM_JSON"
fi
