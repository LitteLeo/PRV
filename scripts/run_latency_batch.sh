#!/usr/bin/env bash
# PRV Latency 实验批量脚本（对齐 DynamicRAG 论文延迟实验）
#
# 维度1：LLM 调用次数 vs NQ EM（固定 20 篇文档，no-planning / planning 两种配置）
# 维度2：三类 PRV 相关场景延迟（仅重排 top-20、point-wise×20、PRV 单轮 top-20），每场景 3 次取平均（Vanilla 见论文）
#
# 运行前准备：
#   1. 启动 vLLM：vllm serve /path/to/REAP-all-merged --host 0.0.0.0 --port 8000
#   2. 数据目录：export DATA_DIR=/path/to/eval_data  # 需含 nq.jsonl
#
# 运行（在 PRV 目录下）：
#   bash scripts/run_latency_batch.sh
#
# 可选环境变量：
#   DATA_DIR         默认 /home/lfy/data/eval_data
#   OUT_DIR          结果目录，默认 experiments/PRV/latency
#   N_SAMPLES        采样条数，默认 50
#   N_RUNS           维度2 每场景运行次数，默认 3
#   DIM              只跑维度1/2：export DIM=1 或 DIM=2
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
OUT_DIR="${OUT_DIR:-$BASE/../experiments/PRV/latency}"
N_SAMPLES="${N_SAMPLES:-50}"
N_RUNS="${N_RUNS:-3}"
DIM="${DIM:-both}"

mkdir -p "$OUT_DIR"
IN="${DATA_DIR}/nq.jsonl"

# 检查 vLLM
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
if ! curl -sf --connect-timeout 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "Error: vLLM not reachable at http://${VLLM_HOST}:${VLLM_PORT}"
  echo "Start first: vllm serve /path/to/REAP-all-merged --host 0.0.0.0 --port ${VLLM_PORT}"
  exit 1
fi
echo "vLLM OK at http://${VLLM_HOST}:${VLLM_PORT}"

if [ ! -f "$IN" ]; then
  echo "Error: NQ input not found: $IN"
  echo "Set DATA_DIR to a directory containing nq.jsonl"
  exit 1
fi

echo "[Latency] Running dim=$DIM, n_samples=$N_SAMPLES, n_runs=$N_RUNS, output=$OUT_DIR"
python "$BASE/evaluation/run_latency_experiment.py" \
  --input-jsonl "$IN" \
  --output-dir "$OUT_DIR" \
  --dim "$DIM" \
  --n-samples "$N_SAMPLES" \
  --n-runs "$N_RUNS" \
  --seed 42

echo "[DONE] Latency results in $OUT_DIR (dim1_summary.json, dim2_summary.json)"
