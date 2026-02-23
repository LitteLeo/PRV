#!/usr/bin/env bash
# REAP 基线批处理：在 eval_data 上与 PRV 同格式的数据集上跑 REAP（E5-on-ctxs + plan + extract + answer），产出 EM 与效率画像。
#
# 用途：对比实验「REAP + 重排 + 统一模型」中增加一行 REAP 基线；与 PRV 同输入同评估。
#
# 用法：
#   cd PRV && bash scripts/run_reap_baseline_batch.sh
#
# 环境变量：
#   DATA_DIR          默认 /home/lfy/data/eval_data（与 run_prv_* 一致）
#   SAMPLE_SIZE       每数据集采样条数（默认 100）
#   REAP_EVAL_TOP_K   ctxs 上 E5 取 top-k（默认 40）
#   TRACK_VRAM        设为 1 则记录 client 侧峰值显存（可选）
#
set -euo pipefail

BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/lfy/data/eval_data}"
EXP_DIR="${EXP_DIR:-$BASE/../experiments/REAP_baseline}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/results}"
LOG_DIR="${LOG_DIR:-$EXP_DIR/logs}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
REAP_EVAL_TOP_K="${REAP_EVAL_TOP_K:-40}"
TRACK_VRAM="${TRACK_VRAM:-0}"

# 与 PRV efficiency_profile 批处理保持一致的数据集列表
DATASETS=(
  "nq.jsonl"
  "2wikimqa.jsonl"
  "hotpotqa.jsonl"
  "eli5.jsonl"
  "fever.jsonl"
  "arc_challenge.jsonl"
  "asqa_eval_gtr_top100.jsonl"
  "popqa.jsonl"
)

mkdir -p "$OUT_DIR" "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M)"
RUN_ID="reap_baseline_${ts}"
RUN_DIR="${EXP_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}/results" "${RUN_DIR}/logs"

echo "REAP baseline batch: sample_size=$SAMPLE_SIZE, eval_top_k=$REAP_EVAL_TOP_K, track_vram=$TRACK_VRAM"

VRAM_ARGS=()
[ "$TRACK_VRAM" = "1" ] && VRAM_ARGS=(--track-vram)

for f in "${DATASETS[@]}"; do
  IN="${DATA_DIR}/${f}"
  base="${f%.jsonl}"
  OUT_JSON="${OUT_DIR}/reap_baseline_${base}_n${SAMPLE_SIZE}.json"
  LOG_FILE="${LOG_DIR}/reap_baseline_${base}_n${SAMPLE_SIZE}_${ts}.log"

  if [ ! -f "$IN" ]; then
    echo "[WARN] skip (missing): $IN"
    continue
  fi

  echo "[RUN] $base (n=$SAMPLE_SIZE)" | tee -a "$LOG_FILE"
  python "$BASE/evaluation/run_reap_baseline_eval.py" \
    --input-jsonl "$IN" \
    --output-json "$OUT_JSON" \
    --eval-top-k "$REAP_EVAL_TOP_K" \
    --sample-size "$SAMPLE_SIZE" \
    --shuffle \
    --seed 42 \
    "${VRAM_ARGS[@]}" \
    >> "$LOG_FILE" 2>&1

  PROFILE_JSON="${OUT_JSON%.json}_efficiency_profile.json"
  [ -s "$PROFILE_JSON" ] && cp -f "$PROFILE_JSON" "${RUN_DIR}/results/" 2>/dev/null || true
  [ -s "$OUT_JSON" ] && cp -f "$OUT_JSON" "${RUN_DIR}/results/" 2>/dev/null || true
done

echo ""
echo "--- REAP baseline efficiency profiles ---"
for f in "${DATASETS[@]}"; do
  base="${f%.jsonl}"
  PROFILE_JSON="${OUT_DIR}/reap_baseline_${base}_n${SAMPLE_SIZE}_efficiency_profile.json"
  [ -s "$PROFILE_JSON" ] && echo "  $base: $PROFILE_JSON"
done
echo "[DONE] REAP baseline results in ${RUN_DIR}"
