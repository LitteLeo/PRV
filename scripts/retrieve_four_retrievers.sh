#!/usr/bin/env bash
# 用四检索器（DPR、Contriever、MonoT5、E5）在 Wikipedia 上检索，生成 12 个 JSONL，每 benchmark 抽样 50（seed 42）。
# 生成后运行: bash scripts/run_prv_four_retrievers_batch.sh
#
# 试跑一条（只跑 nq + Contriever，2 条样本）：TEST=1 bash scripts/retrieve_four_retrievers.sh
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
OUT_DIR="${OUT_DIR:-$DATA_DIR}"
MODEL_DIR="${MODEL_DIR:-/home/lfy/projects/models}"
PASSAGES="${PASSAGES:-/home/lfy/data/wikipedia_embeddings/psgs_w100.tsv}"
# Contriever 预计算 embeddings（Wikipedia）
CONTRIEVER_EMBEDDINGS="${CONTRIEVER_EMBEDDINGS:-/home/lfy/data/wikipedia_embeddings/wikipedia_embeddings/passages_*}"
# E5 Wikipedia 索引目录（需先用 generate_index.py 对 psgs_w100 建好 E5 索引）
E5_INDEX_DIR="${E5_INDEX_DIR:-/home/lfy/data/wikipedia_embeddings/e5_index}"
# DPR Wikipedia 索引目录（可选，无则脚本内用 DPR 现场编码建索引，较慢）
DPR_INDEX_DIR="${DPR_INDEX_DIR:-}"

# 试跑：只跑 nq + Contriever，2 条样本
if [ "${TEST:-0}" = "1" ]; then
  SAMPLE_SIZE="${SAMPLE_SIZE:-2}"
  echo "[TEST] 试跑：仅 nq + Contriever，${SAMPLE_SIZE} 条"
fi
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
SEED=42
TOP_K=40
DYNAMICRAG_DIR="${DYNAMICRAG_DIR:-$(cd "$BASE/../DynamicRAG" 2>/dev/null && pwd)}"

mkdir -p "$OUT_DIR"
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# benchmark 名 -> 源 jsonl 文件名
get_source_file() { case "$1" in nq) echo "nq.jsonl";; hotpotqa) echo "hotpotqa.jsonl";; asqa) echo "asqa_eval_gtr_top100.jsonl";; *) echo "";; esac }
get_output_file() { local b=$1 r=$2; [ "$b" = "asqa" ] && echo "asqa_eval_gtr_top100_${r}.jsonl" || echo "${b}_${r}.jsonl"; }

# ----- Contriever：调用 DynamicRAG retriever.py -----
run_contriever() {
  local bench=$1
  local src="$DATA_DIR/$(get_source_file "$bench")"
  local out="$OUT_DIR/$(get_output_file "$bench" contriever)"
  [ ! -f "$src" ] && echo "[WARN] skip Contriever $bench: no $src" && return 0
  if [ -s "$out" ]; then echo "[SKIP] Contriever $bench: $out exists"; return 0; fi
  local tmp="$TMP_DIR/${bench}_contriever_query.jsonl"
  python3 -c "
import json, random
random.seed($SEED)
with open('$src') as f: lines = [l for l in f if l.strip()]
n = min($SAMPLE_SIZE, len(lines))
sampled = random.sample(lines, n)
with open('$tmp','w') as f:
  for L in sampled:
    o = json.loads(L)
    o['retrieved_question'] = o.get('question','')
    f.write(json.dumps(o, ensure_ascii=False)+'\n')
"
  [ ! -d "$DYNAMICRAG_DIR" ] && echo "[WARN] DynamicRAG not found, skip Contriever" && return 0
  echo "[RUN] Contriever $bench -> $out"
  cd "$DYNAMICRAG_DIR"
  python retriever.py \
    --model_name_or_path "$MODEL_DIR/contriever" \
    --passages "$PASSAGES" \
    --passages_embeddings "$CONTRIEVER_EMBEDDINGS" \
    --query "$tmp" \
    --output_dir "$out" \
    --n_docs "$TOP_K" \
    --save_or_load_index \
    --projection_size 768 --n_subquantizers 0 --n_bits 8
  cd - >/dev/null
}

# ----- DPR / MonoT5 / E5：统一用 Python 脚本 -----
run_py_retriever() {
  local retriever=$1
  local bench=$2
  local src="$DATA_DIR/$(get_source_file "$bench")"
  local out="$OUT_DIR/$(get_output_file "$bench" "$retriever")"
  [ ! -f "$src" ] && echo "[WARN] skip $retriever $bench: no $src" && return 0
  if [ -s "$out" ]; then echo "[SKIP] $retriever $bench: $out exists"; return 0; fi
  echo "[RUN] $retriever $bench -> $out"
  python "$BASE/scripts/retrieve_wikipedia.py" \
    --retriever "$retriever" \
    --benchmark "$bench" \
    --questions-jsonl "$src" \
    --output "$out" \
    --model-dir "$MODEL_DIR" \
    --passages "$PASSAGES" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --top-k "$TOP_K" \
    --e5-index-dir "${E5_INDEX_DIR:-}" \
    --dpr-index-dir "${DPR_INDEX_DIR:-}" \
    --contriever-embeddings "${CONTRIEVER_EMBEDDINGS:-}" \
    --dynamicrag-dir "${DYNAMICRAG_DIR:-}"
}

if [ "${TEST:-0}" = "1" ]; then
  run_contriever "nq"
else
  for bench in nq hotpotqa asqa; do
    run_contriever "$bench"
  done
  for retriever in dpr monot5 e5; do
    for bench in nq hotpotqa asqa; do
      run_py_retriever "$retriever" "$bench"
    done
  done
fi

echo "[DONE] Retrieval outputs in $OUT_DIR (*_dpr.jsonl, *_contriever.jsonl, *_monot5.jsonl, *_e5.jsonl)"
echo "Then run: cd $BASE && bash scripts/run_prv_four_retrievers_batch.sh"
