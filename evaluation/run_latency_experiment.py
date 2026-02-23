"""
PRV Latency 实验脚本（对齐 DynamicRAG 论文延迟实验设置）

两个维度：
- 维度1（LLM 调用次数 vs 效果）：固定检索 20 篇文档，对比不同配置下的「最大 LLM 调用次数」与 NQ 平均 EM。
- 维度2（Token/延迟）：多场景单轮推理耗时，每场景跑 n_runs 次取平均延迟（秒）。

场景对应关系（维度2，仅 PRV 相关；Vanilla 基线见论文）：
- rerank_only_top20：Question + Top-20 Docs，仅重排序 1 次前向
- pointwise_20：(Question + Single Doc)×20，模拟 point-wise 逐篇打分，20 次前向
- prv_single_top20：PRV 单轮（重排 + 事实提取 + 合成），Top-20 输入

控制变量建议（与论文对齐）：推理框架/精度一致；延迟取 n_runs 次平均；NQ 数据固定 seed 采样。
运行前：启动 vLLM，设置 DATA_DIR；NQ 数据为 DATA_DIR/nq.jsonl。
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib import prv_reranker

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from run_prv_eval import ctxs_to_search_hits, run_prv_eval_single
from evaluation_script import _exact_match_score


def load_entries(path: str, n_samples: int = None, seed: int = None):
    with open(path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if n_samples is not None and n_samples > 0:
        rng = random.Random(seed)
        entries = rng.sample(entries, min(n_samples, len(entries)))
    return entries


def get_answers_from_entry(entry):
    answers = entry.get("answers", [])
    out = []
    for a in answers:
        if isinstance(a, str):
            out.append(a.strip())
        elif isinstance(a, dict):
            out.append((a.get("text") or a.get("answer") or "").strip())
        else:
            out.append(str(a).strip())
    return [x for x in out if x]


def compute_em_over_result(original_entries_by_id: dict, result_list: list) -> float:
    """result_list 为 run_prv_eval 输出的 list，每项含 id, response (list)。"""
    correct, total = 0, 0
    for item in result_list:
        uid = item.get("id")
        golds = original_entries_by_id.get(uid)
        if not golds:
            continue
        pred = (item.get("response") or [""])[0]
        total += 1
        if any(_exact_match_score(pred, g) for g in golds):
            correct += 1
    return (correct / total) if total else 0.0


def run_dim1(input_jsonl: str, output_dir: str, n_samples: int, seed: int):
    """
    维度1：跑两种配置（no-planning / planning），固定 20 篇文档，收集 LLM 调用次数与 EM。
    输出 dim1_summary.json：各配置的 avg_llm_calls, em。
    """
    os.makedirs(output_dir, exist_ok=True)
    entries = load_entries(input_jsonl, n_samples=n_samples, seed=seed)
    original_by_id = {e.get("id"): get_answers_from_entry(e) for e in entries}

    # 写临时 input（固定 20 篇）
    temp_input = os.path.join(output_dir, "dim1_input.jsonl")
    with open(temp_input, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    configs = [
        ("no_planning_top20", ["--no-planning", "--max-docs", "20"]),
        ("planning_top20", ["--eval-top-k", "20"]),
    ]
    summary = []
    for name, extra_args in configs:
        out_json = os.path.join(output_dir, f"dim1_{name}.json")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "evaluation", "run_prv_eval.py"),
            "--input-jsonl", temp_input,
            "--output-json", out_json,
        ] + extra_args
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        with open(out_json, "r", encoding="utf-8") as f:
            result_list = json.load(f)
        total_calls = sum((x.get("llm_stats") or {}).get("total_llm_calls", 0) for x in result_list)
        avg_calls = total_calls / len(result_list) if result_list else 0
        em = compute_em_over_result(original_by_id, result_list)
        summary.append({
            "config": name,
            "avg_llm_calls": round(avg_calls, 2),
            "em": round(em, 4),
            "n": len(result_list),
        })
    out_summary = os.path.join(output_dir, "dim1_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Dim1] Wrote {out_summary}")
    return summary


def run_dim2(input_jsonl: str, output_dir: str, n_samples: int, n_runs: int, seed: int):
    """
    维度2：四类场景各跑 n_runs 次，每次对 n_samples 条取平均延迟，再对 n_runs 取均值与标准差。
    场景：vanilla_top10, rerank_only_top20, pointwise_20, prv_single_top20
    """
    os.makedirs(output_dir, exist_ok=True)
    entries = load_entries(input_jsonl, n_samples=n_samples, seed=seed)
    llm_adapter.configure_llm_provider()

    # 确保重排开启（pointwise / prv_single 需要）
    if not getattr(config, "USE_PRV_RERANK", True):
        config.USE_PRV_RERANK = True

    def ctxs_to_hits(ctxs, k=20):
        return ctxs_to_search_hits(ctxs, max_docs=k)

    # 仅跑 PRV 相关场景（Vanilla 基线论文已给）
    scenarios = [
        ("rerank_only_top20", lambda q, ctxs: prv_reranker.rerank_documents(q, ctxs_to_hits(ctxs, 20))),
        ("pointwise_20", lambda q, ctxs: [
            prv_reranker.rerank_documents(q, [ctxs_to_hits(ctxs, 20)[i]]) for i in range(min(20, len(ctxs)))
        ]),
        ("prv_single_top20", lambda q, ctxs: run_prv_eval_single(q, ctxs, max_docs=20, is_short_phrase=True)),
    ]

    results = {}
    for scenario_name, run_one in scenarios:
        run_latencies = []  # 每轮（整次遍历）的平均延迟
        for run_idx in range(n_runs):
            latencies = []
            for entry in entries:
                question = entry.get("question", "")
                ctxs = entry.get("ctxs", entry.get("docs", []))
                if not question or not ctxs:
                    continue
                t0 = time.perf_counter()
                try:
                    run_one(question, ctxs)
                except Exception:
                    pass
                latencies.append(time.perf_counter() - t0)
            run_latencies.append(sum(latencies) / len(latencies) if latencies else 0)
        mean_lat = sum(run_latencies) / len(run_latencies)
        var = sum((x - mean_lat) ** 2 for x in run_latencies) / len(run_latencies) if run_latencies else 0
        std_lat = var ** 0.5
        results[scenario_name] = {
            "latency_mean_s": round(mean_lat, 4),
            "latency_std_s": round(std_lat, 4),
            "n_runs": n_runs,
            "n_samples": len(entries),
        }
        print(f"[Dim2] {scenario_name}: mean={mean_lat:.4f}s std={std_lat:.4f}s")

    out_path = os.path.join(output_dir, "dim2_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Dim2] Wrote {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="PRV Latency experiment (dim1: LLM calls vs EM; dim2: latency by scenario)")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL (e.g. NQ: question, ctxs, answers, id)")
    parser.add_argument("--output-dir", required=True, help="Output directory for dim1/dim2 summaries and temp files")
    parser.add_argument("--dim", choices=["1", "2", "both"], default="both", help="Which dimension to run")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples (default 50)")
    parser.add_argument("--n-runs", type=int, default=3, help="For dim2: number of runs per scenario (default 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    if args.dim in ("1", "both"):
        run_dim1(args.input_jsonl, args.output_dir, args.n_samples, args.seed)
    if args.dim in ("2", "both"):
        run_dim2(args.input_jsonl, args.output_dir, args.n_samples, args.n_runs, args.seed)


if __name__ == "__main__":
    main()
