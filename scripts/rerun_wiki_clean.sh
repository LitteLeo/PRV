#!/usr/bin/env bash
set -e
set +u

BASE_DIR="/home/lfy/projects/REAP"
RESULTS_DIR="${BASE_DIR}/experiments"
DATASET_NAME="2wikimqa"

echo "=========================================="
echo "Cleaning and rerunning: $DATASET_NAME"
echo "=========================================="

# 1. 清理之前的预测结果
echo "[1/4] Cleaning previous results..."
rm -f "${RESULTS_DIR}/${DATASET_NAME}_predictions.jsonl"
rm -f "${RESULTS_DIR}/${DATASET_NAME}_predictions_*.jsonl"
rm -f "${RESULTS_DIR}/${DATASET_NAME}_metrics.txt"
rm -f "${RESULTS_DIR}/${DATASET_NAME}_llm_stats.json"
echo "  Previous results cleaned"

# 2. 停止并重启服务
echo "[2/4] Restarting services..."

# 停止旧服务
pkill -f "vllm serve" || true
pkill -f "uvicorn.*start_e5_server" || true
sleep 5

# 清理 GPU
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true

# 启动 E5 服务（后台）
echo "  Starting E5 service..."
cd "$BASE_DIR"
nohup python -m uvicorn search.start_e5_server_main:app --host 0.0.0.0 --port 8090 > /tmp/e5_service.log 2>&1 &
E5_PID=$!
echo "  E5 service started (PID: $E5_PID)"
sleep 5

# 启动 vLLM 服务（后台）
echo "  Starting vLLM service..."
nohup vllm serve /home/lfy/projects/models/REAP-all \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85 > /tmp/vllm_service.log 2>&1 &
VLLM_PID=$!
echo "  vLLM service started (PID: $VLLM_PID)"
echo "  Waiting for vLLM to initialize (30 seconds)..."
sleep 30

# 3. 测试服务
echo "[3/4] Testing services..."
curl -s http://127.0.0.1:8000/v1/models > /dev/null && echo "  ✓ vLLM service OK" || echo "  ✗ vLLM service not ready"
curl -s -X POST http://127.0.0.1:8090/ -H "Content-Type: application/json" -d '{"query":"test","k":1}' > /dev/null && echo "  ✓ E5 service OK" || echo "  ✗ E5 service not ready"

# 4. 运行推理
echo "[4/4] Running inference..."
cd "$BASE_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate reap 2>/dev/null || true

python evaluation/generate_predictions_from_multistep.py \
    /home/lfy/data/eval_data/2wikimqa_reap_format.jsonl \
    "${RESULTS_DIR}/${DATASET_NAME}_predictions.jsonl" \
    --max_workers 2

echo ""
echo "=========================================="
echo "Inference completed!"
echo "Results: ${RESULTS_DIR}/${DATASET_NAME}_predictions.jsonl"
echo "=========================================="

# 5. 分析LLM统计（如果脚本存在）
if [ -f "${BASE_DIR}/scripts/analyze_llm_stats.py" ]; then
    echo ""
    echo "Analyzing LLM statistics..."
    python "${BASE_DIR}/scripts/analyze_llm_stats.py" \
        "${RESULTS_DIR}/${DATASET_NAME}_predictions.jsonl" \
        --output "${RESULTS_DIR}/${DATASET_NAME}_llm_stats.json"
fi
