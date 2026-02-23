"""
REAP 基线评测：在 eval_data（与 PRV 同格式：question + ctxs + answers）上跑 REAP 全流程。

用途：对比实验「REAP + 重排 + 统一模型」中增加一行 REAP 基线。
- 检索：仅用当前题 ctxs，E5 在 ctxs 上做相似度排序（与 PRV 评测一致，不查外部 E5 大库）。
- 模型/端口：LLM 配置来自 **PRV 的 config**（本脚本先加入 PRV 到 sys.path，REAP 的 llm 代码 import config 时会读到 PRV/config.py）。即 VLLM_LLM_MODEL、VLLM_LLM_PORT（及若 VLLM_USE_DEDICATED_MODELS=True 则各任务端口）均以 PRV/config.py 为准。启动 vLLM 时需与 PRV config 一致，否则 404。
- 输出：与 run_prv_eval 同格式的 JSON + efficiency_profile，便于与 PRV 对比 EM 与显存/耗时。

显存说明：若使用 vLLM 独立进程部署，Peak VRAM 在服务端；本脚本记录的 peak_vram_mb 为评测进程
（含 E5 编码器）的 GPU 占用。三模型 vs 单模型显存对比见 PRV/docs/VRAM_COMPARISON.md。
"""
import argparse
import json
import os
import random
import sys
import time
from contextvars import ContextVar

from tqdm import tqdm

# 先加入 PRV 根与脚本目录，以便 eval_retrieve、evaluation_script、config(E5) 来自 PRV
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import eval_retrieve
from evaluation_script import _exact_match_score

# 再加入 REAP 根（与 PRV 同级的 REAP 目录），使后续 rag_pipeline_lib 来自 REAP
REAP_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "REAP")
if os.path.isdir(REAP_ROOT) and REAP_ROOT not in sys.path:
    sys.path.insert(0, REAP_ROOT)
# 确保从 REAP 导入（REAP 的 rag_pipeline_lib 会覆盖 PRV 的，因为后插入）
from rag_pipeline_lib import core as reap_core
from rag_pipeline_lib import pipeline as reap_pipeline
from rag_pipeline_lib import llm_adapter as reap_llm

# 当前评测条的 ctxs，供 patch 后的 retrieve_context 使用
current_eval_ctxs_ctx: ContextVar = ContextVar("current_eval_ctxs", default=None)

# E5 编码器（PRV 的 eval_retrieve 在 ctxs 上排序用），脚本内懒加载
_e5_encoder = None
EVAL_TOP_K = 40


def _get_e5_encoder():
    global _e5_encoder
    if _e5_encoder is None:
        _e5_encoder = eval_retrieve.get_e5_encoder()
    return _e5_encoder


def _reap_retrieve_context_for_eval(query: str):
    """替代 REAP core.retrieve_context：在当期题的 ctxs 上用 E5 排序取 top-k，不查外部检索服务。"""
    ctxs = current_eval_ctxs_ctx.get()
    if not ctxs:
        return []
    encoder = _get_e5_encoder()
    return eval_retrieve.rank_ctxs_by_query(query, ctxs, k=EVAL_TOP_K, encoder=encoder)


def run_reap_baseline_eval(
    question: str,
    ctxs: list,
    tracer=None,
    verbose: bool = False,
) -> str:
    """单条：设置当前 ctxs，运行 REAP 多步 pipeline，返回最终答案。"""
    current_eval_ctxs_ctx.set(ctxs)
    try:
        answer = reap_pipeline.run_multistep_pipeline(question, verbose=verbose, trace_collector=tracer)
        return (answer or "").strip()
    finally:
        current_eval_ctxs_ctx.set(None)


def main():
    parser = argparse.ArgumentParser(
        description="REAP baseline on eval_data (question+ctxs), same format as PRV eval."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL (question, ctxs, answers, id...)")
    parser.add_argument("--output-json", required=True, help="Output JSON (list of items with response)")
    parser.add_argument("--eval-top-k", type=int, default=40, help="E5 top-k within ctxs (default 40)")
    parser.add_argument("--sample-size", type=int, default=None, help="Only run N samples")
    parser.add_argument("--shuffle", action="store_true", help="With --sample-size: random sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --shuffle")
    parser.add_argument("--verbose", action="store_true", help="Per-item progress")
    parser.add_argument("--track-vram", action="store_true", help="Record peak VRAM (client-side, torch.cuda)")
    args = parser.parse_args()

    global EVAL_TOP_K
    EVAL_TOP_K = args.eval_top_k

    # Patch REAP 的 retrieve_context 为「仅 ctxs + E5 排序」
    original_retrieve = reap_core.retrieve_context
    reap_core.retrieve_context = _reap_retrieve_context_for_eval

    try:
        print("Configuring REAP LLM provider...")
        reap_llm.configure_llm_provider()
        print("Pre-loading E5 encoder (PRV eval_retrieve)...")
        _get_e5_encoder()
    finally:
        pass

    print("Loading input...")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if args.sample_size is not None and args.sample_size > 0:
        n = min(args.sample_size, len(entries))
        if args.shuffle:
            rng = random.Random(args.seed)
            entries = rng.sample(entries, n)
            print(f"Using random sample of {n} (seed={args.seed}).")
        else:
            entries = entries[:n]
            print(f"Using first {n} samples.")

    try:
        torch = __import__("torch")
        has_cuda = getattr(torch, "cuda", None) and torch.cuda.is_available()
    except Exception:
        has_cuda = False

    result = []
    for entry in tqdm(entries, desc="REAP baseline eval"):
        question = entry.get("question", "")
        ctxs = entry.get("docs", entry.get("ctxs", []))
        if not question:
            entry["response"] = [""]
            entry["latency_s"] = 0.0
            entry["llm_stats"] = {"total_llm_calls": 0, "iterations": 0, "call_breakdown": {}}
            entry["peak_vram_mb"] = None
            result.append(entry)
            continue

        tracer = reap_pipeline.Tracer()
        reap_pipeline.tracer_context.set(tracer)

        if args.track_vram and has_cuda:
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        try:
            answer = run_reap_baseline_eval(question, ctxs, tracer=tracer, verbose=args.verbose)
            tracer.commit_pending()
        except Exception as e:
            tracer.commit_pending()
            answer = f"Error: {e}"
        finally:
            latency = time.perf_counter() - t_start

        call_breakdown = {}
        for log_entry in tracer.log:
            name = log_entry.get("adapter_function_name", "unknown")
            call_breakdown[name] = call_breakdown.get(name, 0) + 1

        entry["latency_s"] = latency
        entry["llm_stats"] = {
            "total_llm_calls": len(tracer.log),
            "iterations": getattr(tracer, "iteration_count", 0),
            "call_breakdown": call_breakdown,
        }
        entry["response"] = [answer]
        if args.track_vram and has_cuda:
            entry["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        else:
            entry["peak_vram_mb"] = None

        golds = entry.get("answers", entry.get("answer", []))
        if not isinstance(golds, list):
            golds = [golds] if golds else []
        pred = answer.strip() if answer else ""
        entry["correct"] = any(_exact_match_score(pred, g) for g in golds) if golds else None
        result.append(entry)

    # 恢复原始 retrieve_context（便于其他脚本复用）
    reap_core.retrieve_context = original_retrieve

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(result)} items to {args.output_json}")

    # ---------- 效率画像（与 PRV efficiency_profile 对齐字段）----------
    n = len(result)
    correct_all = sum(1 for r in result if r.get("correct") is True)
    total_wall = sum(r.get("latency_s") or 0 for r in result)
    qps = (n / total_wall) if total_wall and n else None
    peak_vram_list = [r.get("peak_vram_mb") for r in result if r.get("peak_vram_mb") is not None]
    peak_vram_max = max(peak_vram_list) if peak_vram_list else None

    efficiency_profile = {
        "baseline": "REAP",
        "eval_top_k": args.eval_top_k,
        "total_samples": n,
        "em_overall_pct": round(correct_all / n * 100, 2) if n else 0,
        "total_wall_clock_s": round(total_wall, 2),
        "qps": round(qps, 4) if qps is not None else None,
        "peak_vram_mb": peak_vram_max,
        "note_peak_vram": "Client-side (E5 encoder). For server-side (vLLM) peak, use nvidia-smi or run_vram_comparison.",
    }
    profile_path = args.output_json.rsplit(".", 1)[0] + "_efficiency_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(efficiency_profile, f, ensure_ascii=False, indent=2)
    print("\n--- REAP Baseline Efficiency Profile ---")
    print(f"  EM: {efficiency_profile['em_overall_pct']}%")
    print(f"  Wall-clock: {efficiency_profile['total_wall_clock_s']} s   QPS: {efficiency_profile.get('qps')}")
    if peak_vram_max is not None:
        print(f"  Peak VRAM (client): {peak_vram_max} MB")
    print(f"  Profile saved to {profile_path}")


if __name__ == "__main__":
    main()
