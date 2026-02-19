#!/usr/bin/env bash
set -euo pipefail
set =u

# 配置
BASE_DIR="/home/lfy/projects/REAP"
DATA_DIR="/home/lfy/data/eval_data"
CORPUS_DIR="/home/lfy/data/corpus"
EMBEDDINGS_DIR="/home/lfy/data/corpus_embeddings"
RESULTS_DIR="/home/lfy/projects/REAP/experiments"
MODEL_PATH="/home/lfy/projects/models/e5-large-v2"

# 创建目录
mkdir -p "$CORPUS_DIR" "$EMBEDDINGS_DIR" "$RESULTS_DIR"

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

cd "$BASE_DIR"

# 激活环境
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate reap

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
    
    # 1. 转换数据格式 + 提取语料库
    if [ ! -f "$reap_format" ]; then
        echo "[1/4] Converting dataset format..."
        python scripts/prepare_dataset.py \
            --input_file "$input_file" \
            --output_file "$reap_format" \
            --corpus_output "$corpus_file"
    else
        echo "[1/4] Converted format already exists, skipping..."
    fi
    
    # 2. 生成向量索引（如果语料库存在且索引不存在）
    if [ -f "$corpus_file" ] && [ ! -d "$embeddings_dir" ] || [ -z "$(ls -A $embeddings_dir 2>/dev/null)" ]; then
        echo "[2/4] Generating embeddings index..."
        python scripts/generate_index.py \
            --corpus_file "$corpus_file" \
            --output_dir "$embeddings_dir" \
            --corpus_data_file "${CORPUS_DIR}/${dataset_name}_corpus.pkl" \
            --model_path "$MODEL_PATH" \
            --shard_size 50000 \
            --device cuda:0
    else
        echo "[2/4] Embeddings index already exists, skipping..."
    fi
    
    # 3. 运行推理（如果预测文件不存在）
    if [ ! -f "$predictions_file" ] || [ ! -s "$predictions_file" ]; then
        echo "[3/4] Running REAP inference..."
        python evaluation/generate_predictions_from_multistep.py \
            "$reap_format" \
            "$predictions_file" \
            --max_workers 4
    else
        echo "[3/4] Predictions already exist, skipping..."
    fi
    
    # 4. 评估（需要原始答案）
    echo "[4/4] Evaluating..."
    # 注意：评估需要原始答案，需要创建一个包含答案的 ground truth 文件
    # 这里先跳过，后面单独处理
    
    echo "Completed: $dataset_name"
    echo ""
done

echo "All datasets processed!"
