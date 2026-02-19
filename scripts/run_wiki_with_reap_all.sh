#!/usr/bin/env bash
set -e
set +u

# 配置
BASE_DIR="/home/lfy/projects/REAP"
DATA_DIR="/home/lfy/data/eval_data"
CORPUS_DIR="/home/lfy/data/corpus"
EMBEDDINGS_DIR="/home/lfy/data/corpus_embeddings"
RESULTS_DIR="/home/lfy/projects/REAP/experiments"
MODEL_PATH="/home/lfy/projects/models/e5-large-v2"
VLLM_MODEL="/home/lfy/projects/models/REAP-all"

DATASET="2wikimqa.jsonl"
DATASET_NAME="2wikimqa"

cd "$BASE_DIR"

# 激活环境
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate reap 2>/dev/null || true

echo "=========================================="
echo "Processing: $DATASET_NAME with REAP-all"
echo "=========================================="

# 1. 转换数据格式 + 提取语料库
INPUT_FILE="${DATA_DIR}/${DATASET}"
REAP_FORMAT="${DATA_DIR}/${DATASET_NAME}_reap_format.jsonl"
CORPUS_FILE="${CORPUS_DIR}/${DATASET_NAME}_corpus.jsonl"

echo "[1/6] Converting dataset format..."
if [ ! -f "$REAP_FORMAT" ]; then
    python scripts/prepare_dataset.py \
        --input_file "$INPUT_FILE" \
        --output_file "$REAP_FORMAT" \
        --corpus_output "$CORPUS_FILE"
else
    echo "  Converted format already exists, skipping..."
fi

# 2. 生成向量索引
EMBEDDINGS_DIR_DATASET="${EMBEDDINGS_DIR}/${DATASET_NAME}"
CORPUS_PKL="${CORPUS_DIR}/${DATASET_NAME}_corpus.pkl"

echo "[2/6] Generating embeddings index..."
if [ -f "$CORPUS_FILE" ] && ([ ! -d "$EMBEDDINGS_DIR_DATASET" ] || [ -z "$(ls -A $EMBEDDINGS_DIR_DATASET 2>/dev/null)" ]); then
    python scripts/generate_index.py \
        --corpus_file "$CORPUS_FILE" \
        --output_dir "$EMBEDDINGS_DIR_DATASET" \
        --corpus_data_file "$CORPUS_PKL" \
        --model_path "$MODEL_PATH" \
        --shard_size 50000 \
        --device cuda:0
else
    echo "  Embeddings index already exists, skipping..."
fi

# 3. 更新配置文件 - 使用 REAP-all
echo "[3/6] Updating config to use REAP-all..."

# 更新 config.py
python << EOF
import re

config_file = "/home/lfy/projects/REAP/config.py"
with open(config_file, 'r') as f:
    content = f.read()

# 替换所有模型路径为 REAP-all
content = re.sub(
    r'VLLM_LLM_MODEL = "[^"]*"',
    'VLLM_LLM_MODEL = "/home/lfy/projects/models/REAP-all"',
    content
)
content = re.sub(
    r'VLLM_ANALYZE_QUERY_MODEL = "[^"]*"',
    'VLLM_ANALYZE_QUERY_MODEL = "/home/lfy/projects/models/REAP-all"',
    content
)
content = re.sub(
    r'VLLM_EXTRACT_FACTS_MODEL = "[^"]*"',
    'VLLM_EXTRACT_FACTS_MODEL = "/home/lfy/projects/models/REAP-all"',
    content
)
content = re.sub(
    r'VLLM_UPDATE_PLAN_MODEL = "[^"]*"',
    'VLLM_UPDATE_PLAN_MODEL = "/home/lfy/projects/models/REAP-all"',
    content
)
content = re.sub(
    r'VLLM_REPLAN_CONDITIONS_MODEL = "[^"]*"',
    'VLLM_REPLAN_CONDITIONS_MODEL = "/home/lfy/projects/models/REAP-all"',
    content
)
content = re.sub(
    r'VLLM_GENERATE_FINAL_ANSWER_MODEL = "[^"]*"',
    'VLLM_GENERATE_FINAL_ANSWER_MODEL = "/home/lfy/projects/models/REAP-all"',
    content
)

with open(config_file, 'w') as f:
    f.write(content)

print("Config updated to use REAP-all")
EOF

# 4. 更新检索服务配置
echo "[4/6] Updating search service config..."

python << EOF
import re

# 更新 start_e5_server_main.py
server_file = "/home/lfy/projects/REAP/search/start_e5_server_main.py"
with open(server_file, 'r') as f:
    content = f.read()

content = re.sub(
    r"index_dir=['\"].*?['\"]",
    f"index_dir='$EMBEDDINGS_DIR_DATASET/'",
    content
)
content = re.sub(
    r"model_name_or_path=['\"].*?['\"]",
    f"model_name_or_path='$MODEL_PATH'",
    content
)

with open(server_file, 'w') as f:
    f.write(content)

# 更新 e5_searcher.py
searcher_file = "/home/lfy/projects/REAP/search/e5_searcher.py"
with open(searcher_file, 'r') as f:
    content = f.read()

content = re.sub(
    r"pickle_path = ['\"].*?['\"]",
    f"pickle_path = '$CORPUS_PKL'",
    content
)

with open(searcher_file, 'w') as f:
    f.write(content)

print("Search service config updated")
EOF

# 5. 启动服务并运行推理
PREDICTIONS_FILE="${RESULTS_DIR}/${DATASET_NAME}_predictions.jsonl"

echo "[5/6] Starting services and running inference..."
echo "  This will start E5 and vLLM services, run inference, then cleanup after 5 minutes"

python scripts/run_inference_with_services.py \
    --dataset_file "$REAP_FORMAT" \
    --output_file "$PREDICTIONS_FILE" \
    --vllm_model "$VLLM_MODEL" \
    --max_workers 2 \
    --timeout_minutes 5

# 6. 评估
METRICS_FILE="${RESULTS_DIR}/${DATASET_NAME}_metrics.txt"

echo "[6/6] Evaluating results..."
if [ -f "$PREDICTIONS_FILE" ] && [ -s "$PREDICTIONS_FILE" ]; then
    python scripts/evaluate_with_answers.py \
        --original_file "$INPUT_FILE" \
        --predictions_file "$PREDICTIONS_FILE" \
        --output_file "$METRICS_FILE"
    echo ""
    echo "Metrics saved to: $METRICS_FILE"
    cat "$METRICS_FILE"
else
    echo "  No predictions file found, skipping evaluation"
fi

echo ""
echo "=========================================="
echo "Completed: $DATASET_NAME"
echo "=========================================="
echo "Results: $PREDICTIONS_FILE"
echo "Metrics: $METRICS_FILE"
