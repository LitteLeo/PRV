#!/usr/bin/env bash
set -e
set +u  # 允许未绑定变量（解决conda问题）

# 配置
BASE_DIR="/home/lfy/projects/REAP"
DATA_DIR="/home/lfy/data/eval_data"
CORPUS_DIR="/home/lfy/data/corpus"
EMBEDDINGS_DIR="/home/lfy/data/corpus_embeddings"
RESULTS_DIR="/home/lfy/projects/REAP/experiments"
MODEL_PATH="/home/lfy/projects/models/e5-large-v2"
VLLM_MODEL="/home/lfy/projects/models/REAP-plan"

# 数据集列表
DATASETS=(
    "eli5.jsonl"
    "hotpotqa.jsonl"
    "2wikimqa.jsonl"
    "fever.jsonl"
    "triviaqa.jsonl"
    "popqa.jsonl"
    "arc_challenge.jsonl"
    "asqa_eval_gtr_top100.jsonl"
)

# 创建目录
mkdir -p "$CORPUS_DIR" "$EMBEDDINGS_DIR" "$RESULTS_DIR" "$BASE_DIR/scripts/logs"

cd "$BASE_DIR"

# 激活环境
if command -v conda &> /dev/null; then
    source "$HOME/miniconda3/etc/profile.d/cnda.sh" 2>/dev/null || true
    conda activate reap 2>/dev/null || true
fi

# 对每个数据集
for dataset in "${DATASETS[@]}"; do
    dataset_name="${dataset%.jsonl}"
    echo "=========================================="
    echo "Processing: $dataset_name"
    echo "=========================================="
    
    input_file="${DATA_DIR}/${dataset}"
    reap_format="${DATA_DIR}/${dataset_name}_reap_format.jsonl"
    corpus_file="${CORPUS_DIR}/${dataset_name}_corpus.jsonl"
    embeddings_dir="${EMBEDDINGS_DIR}/${dataset_name}"
    predictions_file="${RESULTS_DIR}/${dataset_name}_predictions.jsonl"
    metrics_file="${RESULTS_DIR}/${dataset_name}_metrics.txt"
    log_file="${BASE_DIR}/scripts/logs/${dataset_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # 1. 转换数据格式 + 提取语料库
    if [ ! -f "$reap_format" ]; then
        echo "[1/5] Converting dataset format..."
        python scripts/prepare_dataset.py \
            --input_file "$input_file" \
            --output_file "$reap_format" \
            --corpus_output "$corpus_file" >> "$log_file" 2>&1
    else
        echo "[1/5] Converted format already exists, skipping..."
    fi
    
    # 2. 生成向量索引（如果语料库存在）
    if [ -f "$corpus_file" ] && ([ ! -d "$embeddings_dir" ] || [ -z "$(ls -A $embeddings_dir 2>/dev/null)" ]); then
        echo "[2/5] Generating embeddings index..."
        python scripts/generate_index.py \
            --corpus_file "$corpus_file" \
            --output_dir "$embeddings_dir" \
            --corpus_data_file "${CORPUS_DIR}/${dataset_name}_corpus.pkl" \
            --model_path "$MODEL_PATH" \
            --shard_size 50000 \
            --device cuda:0 >> "$log_file" 2>&1
    else
        echo "[2/5] Embeddings index already exists, skipping..."
    fi
    
    # 3. 更新配置文件（为当前数据集）
    echo "[3/5] Updating config for $dataset_name..."
    python scripts/update_config_for_dataset.py \
        --dataset_name "$dataset_name" \
        --embeddings_dir "$embeddings_dir" \
        --corpus_file "${CORPUS_DIR}/${dataset_name}_corpus.pkl" >> "$log_file" 2>&1
    
    # 4. 启动服务并运行推理
    echo "[4/5] Starting services and running inference..."
    python scripts/run_inference_with_services.py \
        --dataset_file "$reap_format" \
        --output_file "$predictions_file" \
        --vllm_model "$VLLM_MODEL" \
        --max_workers 2 \
        --timeout_minutes 5 >> "$log_file" 2>&1
    
    # 5. 评估
    if [ -f "$predictions_file" ] && [ -s "$predictions_file" ]; then
        echo "[5/5] Evaluating..."
        python scripts/evaluate_with_answers.py \
            --original_file "$input_file" \
            --predictions_file "$predictions_file" \
            --output_file "$metrics_file" >> "$log_file" 2>&1
    fi
    
    echo "Completed: $dataset_name"
    echo ""
done

echo "All datasets processed!"
