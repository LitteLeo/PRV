"""
PRV 评测脚本：使用评测集自带的 (question, ctxs)，与 DynamicRAG 同输入同评估。

- 默认（--use-planning）：完整 PRV = 规划(拆子问题) + 多轮(每轮从 ctxs 做 E5 排序→重排→事实提取→更新计划) → 合成。
  检索仅限当前题 ctxs，用 E5 在 ctxs 上做相似度排序再取 top-k，避免按位置截断把高相关 doc 截掉。
- --no-planning：单轮 ctxs→重排→事实提取→合成（兼容旧行为）。

效率画像（Efficiency Profile）单次运行可产出：
- Fail-safe/解析鲁棒性：ParseIndex 失败率、JSON 解析失败率、fallback 触发率；触发时 vs 未触发时 EM。
- 闭环稳定性：Bridge evidence 保留率、Recovery 成功率（PARTIAL/FAILED 后恢复）、False/未证实事实率。
- 性能：wall-clock、QPS、estimated token（input/output）、输出长度分布（mean/p50/p95）。
- 注：State-decoupling（Rerank with facts vs without）需改 reranker 接口后做对比实验；Parallel vs Serial next_actions 在 pipeline.run_multistep_pipeline(serial_next_actions=True/False) 下各跑一遍对比 wall-clock。
"""
import argparse
import json
import random
import re
import sys
import os
import time

from tqdm import tqdm

# 项目根目录（PRV）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib import prv_reranker
from rag_pipeline_lib import core as rag_core
from rag_pipeline_lib.pipeline import Tracer, tracer_context
from rag_pipeline_lib.efficiency_stats import (
    create_efficiency_stats,
    efficiency_stats_context,
    get_efficiency_stats,
)
# 同目录下的评测检索模块（E5 仅在 ctxs 上排序）
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import eval_retrieve
from evaluation_script import _exact_match_score


def _record_json_attempt():
    s = get_efficiency_stats()
    if s is not None:
        s["json_total"] = s.get("json_total", 0) + 1


def _record_json_fail():
    s = get_efficiency_stats()
    if s is not None:
        s["json_fail"] = s.get("json_fail", 0) + 1
        s["failsafe_triggered"] = True


def ctxs_to_search_hits(ctxs, max_docs=40):
    """将评测集 ctxs（list of {title, text}）转为 PRV 的 search_hits（list of {title, contents}）。仅在不做 E5 排序时用。"""
    hits = []
    for c in ctxs[:max_docs]:
        title = c.get("title", "")
        text = c.get("text", c.get("contents", ""))
        hits.append({"title": title, "contents": text})
    return hits


def build_retrieved_documents_str(search_hits):
    """与 core.retrieve_and_extract_facts 中格式一致。"""
    if not search_hits:
        return "No relevant context found."
    docs_list = []
    for i, hit in enumerate(search_hits):
        content = hit.get("contents", str(hit))
        title = hit.get("title", "")
        if title:
            docs_list.append(f'<document id={i+1} title="{title}">\n{content}\n</document>')
        else:
            docs_list.append(f'<document id={i+1}>\n{content}\n</document>')
    return "\n\n".join(docs_list)


def _rewrite_query_for_failed_retry(question: str, verbose: bool = False) -> str:
    """FAILED 专用：用 LLM 生成一句改写 query 用于再试检索（简化、加关键词等）。失败或空则返回原 question。"""
    if not (question and question.strip()):
        return question or ""
    try:
        prompt = (
            "Rewrite the following search query to improve retrieval: "
            "e.g. simplify terms, add keywords like year/film/release. "
            "Output only the rewritten query, one line, no explanation."
        )
        out = llm_adapter.generate_rag_response(query=prompt, context=question)
        if out and out.strip():
            return out.strip()
    except Exception as e:
        if verbose:
            print(f"  [Rewrite] failed: {e}")
    return question


def run_prv_eval_vanilla_rag(question: str, ctxs: list, top_k: int = 10) -> str:
    """
    Vanilla RAG 基线：无重排、无事实提取，仅用 question + top-k 文档做 1 次 LLM 生成。
    用于 Latency 实验维度2 的「Question + Top-10 Docs」场景。
    """
    search_hits = ctxs_to_search_hits(ctxs, max_docs=top_k)
    if not search_hits:
        return "No relevant context provided."
    context = build_retrieved_documents_str(search_hits)
    answer = llm_adapter.generate_rag_response(question, context)
    return answer.strip() if answer else ""


def run_prv_eval_single(
    question: str,
    ctxs: list,
    max_docs: int = 40,
    verbose: bool = False,
    is_fever: bool = False,
    is_eli5: bool = False,
    is_short_phrase: bool = False,
) -> str:
    """
    单条评测（无规划）：给定 question + ctxs，ctxs 按位置截断 → 重排 → 事实提取 → 答案合成。
    """
    search_hits = ctxs_to_search_hits(ctxs, max_docs=max_docs)
    if not search_hits:
        return "No relevant context provided."

    reranked = prv_reranker.rerank_documents(question, search_hits)
    max_chars = getattr(config, "MAX_DOCUMENT_CHARS", 0)
    if max_chars > 0:
        reranked = rag_core._truncate_hits_to_fit_context(reranked, max_chars)
    retrieved_documents_str = build_retrieved_documents_str(reranked)

    requirement = {"requirement_id": "req1", "question": question, "depends_on": None}
    active_requirement_str = json.dumps(requirement, indent=2)
    known_facts_str = "[]"

    facts_json_str = llm_adapter.extract_facts(
        query=question,
        active_requirement=active_requirement_str,
        known_facts=known_facts_str,
        retrieved_documents=retrieved_documents_str,
    )

    match = re.search(r"\{.*\}", facts_json_str, re.DOTALL)
    if not match:
        _record_json_attempt()
        _record_json_fail()
        collected_facts = {"reasoned_facts": []}
    else:
        _record_json_attempt()
        try:
            extracted = json.loads(match.group(0))
            if "reasoned_facts" not in extracted and isinstance(extracted, dict) and "statement" in extracted:
                collected_facts = {"reasoned_facts": [extracted]}
            else:
                collected_facts = extracted if isinstance(extracted, dict) and "reasoned_facts" in extracted else {"reasoned_facts": []}
        except json.JSONDecodeError:
            _record_json_fail()
            collected_facts = {"reasoned_facts": []}

    s = get_efficiency_stats()
    if s is not None:
        for f in collected_facts.get("reasoned_facts", []):
            s["total_facts"] = s.get("total_facts", 0) + 1
            level = (f.get("fulfillment_level") or "").strip().upper()
            if level == "FAILED_EXTRACT":
                s["failed_facts"] = s.get("failed_facts", 0) + 1
            elif level in ("PARTIAL_CLUE", "PARTIAL"):
                s["partial_facts"] = s.get("partial_facts", 0) + 1

    if is_fever:
        answer = rag_core.synthesize_final_answer_fever(question, collected_facts)
    elif is_eli5:
        answer = rag_core.synthesize_final_answer_paragraph(question, collected_facts)
    elif is_short_phrase:
        answer = rag_core.synthesize_final_answer_short_phrase(question, collected_facts)
    else:
        answer = rag_core.synthesize_final_answer(question, collected_facts)
    return answer.strip() if answer else ""


def run_prv_eval_with_planning(
    question: str,
    ctxs: list,
    eval_top_k: int = 40,
    recovery_top_k: int = None,
    max_iterations: int = 5,
    max_total_attempts: int = 10,
    verbose: bool = False,
    is_fever: bool = False,
    is_eli5: bool = False,
    is_short_phrase: bool = False,
    no_recovery: bool = False,
    planning_only: bool = False,
    use_improved_recovery: bool = False,
) -> str:
    """
    完整 PRV 评测：规划(拆子问题) + 多轮迭代，每轮「从 ctxs 用 E5 排序取 top-k → 重排 → 事实提取 → 更新计划」→ 合成。
    检索仅限当前题 ctxs，不查外部 E5 大库。
    no_recovery: 消融「w/o RePlanner」：PARTIAL/FAILED 时不调用 replan，直接终止并合成。
    planning_only: 消融「Planning-only loop」：多轮规划+检索，但不做结构化事实提取，仅占位推进计划。
    use_improved_recovery: True=Improved 恢复（为 replan 未覆盖的 pending 自动注入再试动作，并用 PARTIAL 线索丰富 query）；False=Origin 原逻辑。
    """
    if not ctxs:
        return "No relevant context provided."

    # 阶段1：查询分解
    analysis_result = None
    for attempt in range(3):
        try:
            analysis_result = rag_core.analyze_and_decompose_query(query=question)
            if analysis_result and analysis_result.get("requirements"):
                break
        except Exception as e:
            if verbose:
                print(f"  analyze attempt {attempt + 1} failed: {e}")
    if not analysis_result or not analysis_result.get("requirements"):
        # 退化为单需求（规划/解析失败；core 已记录 json_fail/failsafe）
        all_requirements = [{"requirement_id": "req1", "question": question, "depends_on": None}]
    else:
        all_requirements = analysis_result["requirements"]

    pending_requirements = list(all_requirements)
    req_id_to_question = {req["requirement_id"]: req["question"] for req in all_requirements}
    collected_facts = {"reasoned_facts": []}
    last_extraction_was_direct_only = True
    iteration_count = 0
    total_attempt_count = 0
    no_progress_rounds = 0
    single_requirement = len(all_requirements) == 1
    # 闭环稳定性：PARTIAL/FAILED 后恢复成功率
    recovery_req_ids_partial_failed = set()

    while iteration_count < max_iterations:
        total_attempt_count += 1
        if total_attempt_count > max_total_attempts:
            break
        if not pending_requirements:
            break

        facts_before = list(collected_facts["reasoned_facts"])
        pending_before = list(pending_requirements)
        req_map_before = dict(req_id_to_question)
        last_direct_before = last_extraction_was_direct_only

        try:
            # SP：update_plan 或 replan_questions（消融 no_recovery 时 PARTIAL/FAILED 不修复计划，直接合成）
            if last_extraction_was_direct_only:
                decision_result = rag_core.update_plan(
                    query=question, collected_facts=collected_facts, pending_requirements=pending_requirements
                )
            elif no_recovery:
                decision_result = {
                    "decision": {"next_step": "SYNTHESIZE_ANSWER", "updated_plan": []}
                }
            else:
                decision_result = rag_core.replan_questions(
                    query=question, collected_facts=collected_facts, pending_requirements=pending_requirements
                )
            if not decision_result or "decision" not in decision_result:
                raise ValueError("Planning step returned no valid decision.")
            decision = decision_result["decision"]

            if decision.get("updated_plan") is not None:
                pending_requirements = decision["updated_plan"]
                req_id_to_question = {req["requirement_id"]: req["question"] for req in pending_requirements}

            next_step = decision.get("next_step")
            if next_step == "SYNTHESIZE_ANSWER":
                break
            if next_step != "CONTINUE_SEARCH":
                raise ValueError(f"Unexpected next_step: {next_step}")

            next_actions = list(decision.get("next_actions") or decision.get("next_questions", []) or [])
            if not next_actions:
                raise ValueError("Planner said CONTINUE_SEARCH but gave no next_actions.")

            # Improved 恢复：为 replan 未覆盖的 pending 注入再试动作，提高 Recovery 成功率
            if use_improved_recovery and next_actions:
                covered_req_ids = {a.get("requirement_id") for a in next_actions if a.get("requirement_id")}
                for req in pending_requirements:
                    req_id = req.get("requirement_id")
                    if not req_id or req_id in covered_req_ids:
                        continue
                    recovery_question = req.get("question") or ""
                    has_partial = False
                    for f in reversed(collected_facts.get("reasoned_facts", [])):
                        if f.get("fulfills_requirement_id") != req_id:
                            continue
                        level = (f.get("fulfillment_level") or "").strip().upper()
                        if level in ("PARTIAL_CLUE", "PARTIAL") and f.get("statement"):
                            recovery_question = f"{recovery_question} [已知部分: {f.get('statement')}]"
                            has_partial = True
                            break
                    # 纯 FAILED（无 PARTIAL）时用 LLM 改写 query 再试
                    if not has_partial:
                        recovery_question = _rewrite_query_for_failed_retry(recovery_question, verbose=verbose)
                    next_actions.append({"requirement_id": req_id, "question": recovery_question})
                    if verbose:
                        print(f"  [Improved recovery] inject retry for {req_id}")

            iteration_new_facts = []
            for action in next_actions:
                if not any(req["requirement_id"] == action.get("requirement_id") for req in pending_requirements):
                    continue
                req = [r for r in pending_requirements if r["requirement_id"] == action.get("requirement_id")][0]
                req_id = action.get("requirement_id")
                # Bridge evidence retention：后续轮所需证据保留率
                dep = req.get("depends_on")
                if dep is not None and str(dep).strip():
                    s = get_efficiency_stats()
                    if s is not None:
                        s["bridge_retention_checks"] = s.get("bridge_retention_checks", 0) + 1
                        if any(f.get("fulfills_requirement_id") == dep for f in collected_facts.get("reasoned_facts", [])):
                            s["bridge_retention_ok"] = s.get("bridge_retention_ok", 0) + 1
                if planning_only:
                    # 消融 Planning-only：不做结构化事实提取，仅占位使计划推进
                    iteration_new_facts.append({
                        "fulfills_requirement_id": req_id,
                        "statement": "Retrieved.",
                        "fulfillment_level": "DIRECT_ANSWER",
                    })
                    continue
                sub_query = action.get("question") or req["question"]
                # 恢复轮扩大检索窗口（不超过 ctxs 长度）；否则用 eval_top_k
                k_recovery = (recovery_top_k or eval_top_k) if req_id in recovery_req_ids_partial_failed else eval_top_k
                k_actual = min(k_recovery, len(ctxs))
                search_hits = eval_retrieve.rank_ctxs_by_query(sub_query, ctxs, k=k_actual)
                extracted = rag_core.extract_facts_given_hits(
                    search_query=sub_query,
                    requirement=req,
                    collected_facts=collected_facts,
                    search_hits=search_hits,
                )
                if isinstance(extracted, dict) and extracted.get("reasoned_facts"):
                    iteration_new_facts.extend(extracted["reasoned_facts"])

            if iteration_new_facts:
                no_progress_rounds = 0
                processed = []
                s = get_efficiency_stats()
                for fact in iteration_new_facts:
                    req_id = fact.get("fulfills_requirement_id")
                    if not req_id:
                        continue
                    level = (fact.get("fulfillment_level") or "").strip().upper()
                    if s is not None:
                        s["total_facts"] = s.get("total_facts", 0) + 1
                        if level == "FAILED_EXTRACT":
                            s["failed_facts"] = s.get("failed_facts", 0) + 1
                        elif level in ("PARTIAL_CLUE", "PARTIAL"):
                            s["partial_facts"] = s.get("partial_facts", 0) + 1
                    if level in ("PARTIAL_CLUE", "PARTIAL", "FAILED_EXTRACT"):
                        recovery_req_ids_partial_failed.add(req_id)
                    q = req_id_to_question.get(req_id, "Unknown")
                    processed.append({
                        "fulfills_requirement_id": req_id,
                        "requirement": q,
                        "reasoning": fact.get("reasoning"),
                        "statement": fact.get("statement"),
                        "fulfillment_level": fact.get("fulfillment_level"),
                    })
                collected_facts["reasoned_facts"].extend(processed)
                fulfilled = {f["fulfills_requirement_id"] for f in iteration_new_facts if f.get("fulfillment_level") == "DIRECT_ANSWER"}
                pending_requirements = [r for r in pending_requirements if r["requirement_id"] not in fulfilled]
                last_extraction_was_direct_only = all(f.get("fulfillment_level") == "DIRECT_ANSWER" for f in iteration_new_facts)
            else:
                last_extraction_was_direct_only = False
                no_progress_rounds += 1

            # 简单启发式早停：单 requirement 且多轮无进展时直接进入合成
            if single_requirement and iteration_count >= 2 and no_progress_rounds >= 1:
                if verbose:
                    print("Heuristic early stop: single requirement with repeated no-progress rounds. Moving to synthesis.")
                break

            iteration_count += 1
        except (json.JSONDecodeError, ValueError, RuntimeError) as e:
            _record_json_fail()
            if verbose:
                print(f"  iteration failed: {e}, rolling back")
            collected_facts["reasoned_facts"] = facts_before
            pending_requirements = pending_before
            req_id_to_question = req_map_before
            last_extraction_was_direct_only = last_direct_before
            continue

    # Recovery success rate：曾 PARTIAL/FAILED 的 req 中，最终得到 DIRECT_ANSWER 的比例
    fulfilled_final = {f["fulfills_requirement_id"] for f in collected_facts.get("reasoned_facts", []) if (f.get("fulfillment_level") or "").strip().upper() == "DIRECT_ANSWER"}
    recovery_ok = len(recovery_req_ids_partial_failed & fulfilled_final)
    recovery_attempts = len(recovery_req_ids_partial_failed)
    st = get_efficiency_stats()
    if st is not None:
        st["recovery_attempts"] = recovery_attempts
        st["recovery_successes"] = recovery_ok

    if is_fever:
        answer = rag_core.synthesize_final_answer_fever(query=question, collected_facts=collected_facts)
    elif is_eli5:
        answer = rag_core.synthesize_final_answer_paragraph(query=question, collected_facts=collected_facts)
    elif is_short_phrase:
        answer = rag_core.synthesize_final_answer_short_phrase(query=question, collected_facts=collected_facts)
    else:
        answer = rag_core.synthesize_final_answer(query=question, collected_facts=collected_facts)
    return answer.strip() if answer else ""


def run_prv_eval_zeroshot(
    question: str,
    is_fever: bool = False,
    is_eli5: bool = False,
    is_short_phrase: bool = False,
) -> str:
    """
    Zero-Shot：无检索文档，仅依赖模型自身知识生成答案。
    用于 Top-N 实验中的 baseline（N=0）。
    """
    collected_facts = {"reasoned_facts": []}
    if is_fever:
        answer = rag_core.synthesize_final_answer_fever(query=question, collected_facts=collected_facts)
    elif is_eli5:
        answer = rag_core.synthesize_final_answer_paragraph(query=question, collected_facts=collected_facts)
    elif is_short_phrase:
        answer = rag_core.synthesize_final_answer_short_phrase(query=question, collected_facts=collected_facts)
    else:
        answer = rag_core.synthesize_final_answer(query=question, collected_facts=collected_facts)
    return answer.strip() if answer else ""


def main():
    parser = argparse.ArgumentParser(description="PRV eval: question+ctxs -> [planning + E5-on-ctxs retrieval] or single-step -> output same format as DynamicRAG")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL (question, ctxs, answers, id...)")
    parser.add_argument("--output-json", required=True, help="Output JSON (list of items with response)")
    parser.add_argument("--use-planning", action="store_true", default=True, help="Full PRV: plan + multi-step with E5-on-ctxs (default)")
    parser.add_argument("--no-planning", action="store_true", help="Single-step: ctxs by position -> rerank -> extract -> synthesize")
    parser.add_argument("--max-docs", type=int, default=40, help="Max ctxs per question for --no-planning (default 40)")
    parser.add_argument("--eval-top-k", type=int, default=40, help="E5 top-k within ctxs for --use-planning (default 40)")
    parser.add_argument("--recovery-top-k", type=int, default=None, help="Larger top-k for recovery rounds; capped by len(ctxs). Omit to not expand (default)")
    parser.add_argument("--sample-size", type=int, default=None, help="Only run N samples (default: all). With --shuffle, random N; else first N.")
    parser.add_argument("--shuffle", action="store_true", help="With --sample-size: sample randomly (use --seed for reproducible subset)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --shuffle")
    parser.add_argument("--verbose", action="store_true", help="Print per-item progress")
    parser.add_argument("--zero-shot", action="store_true", help="No retrieval: answer from model knowledge only (Top-N=0 baseline)")
    parser.add_argument("--no-recovery", action="store_true", help="Ablation: w/o RePlanner, terminate on PARTIAL/FAILED instead of replanning")
    parser.add_argument("--planning-only", action="store_true", help="Ablation: Planning-only loop, no structured fact extraction")
    parser.add_argument("--recovery-mode", choices=("origin", "improved"), default="origin", help="Origin=原逻辑; Improved=为 replan 未覆盖的 pending 注入再试并带 PARTIAL 线索 (default: origin)")
    parser.add_argument("--track-vram", action="store_true", help="记录每样本峰值显存 MB（评测进程侧；vLLM 服务端显存需另用 nvidia-smi 等测量）")
    args = parser.parse_args()
    use_planning = args.use_planning and not args.no_planning
    zero_shot = args.zero_shot

    print("Configuring LLM provider...")
    llm_adapter.configure_llm_provider()
    if zero_shot:
        print("Eval mode: Zero-Shot (no retrieval).")
    elif use_planning:
        mode_parts = ["full PRV (planning + E5-on-ctxs retrieval)"]
        if args.no_recovery:
            mode_parts.append("w/o RePlanner (no recovery)")
        if args.planning_only:
            mode_parts.append("Planning-only loop (no structured facts)")
        if not getattr(config, "USE_VERIFICATION_CONSTRAINTS", True):
            mode_parts.append("w/o Verification constraints")
        if getattr(args, "recovery_mode", "origin") == "improved":
            mode_parts.append("recovery=improved (inject retry + PARTIAL clue + FAILED rewrite)")
        if getattr(args, "recovery_top_k", None) is not None:
            mode_parts.append(f"recovery_top_k={args.recovery_top_k}")
        print("Eval mode: " + "; ".join(mode_parts) + ". Pre-loading E5 encoder...")
        eval_retrieve.get_e5_encoder()
    print("Loading input...")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    input_path_lower = args.input_jsonl.lower()
    is_fever_dataset = "fever" in input_path_lower
    is_eli5_dataset = "eli5" in input_path_lower
    is_short_phrase_dataset = any(
        name in input_path_lower
        for name in ["nq", "hotpotqa", "triviaqa", "2wikimqa", "popqa", "asqa"]
    )
    if args.sample_size is not None and args.sample_size > 0:
        n = min(args.sample_size, len(entries))
        if args.shuffle:
            rng = random.Random(args.seed)
            entries = rng.sample(entries, n)
            print(f"Using random sample of {n} (seed={args.seed}).")
        else:
            entries = entries[:n]
            print(f"Using first {n} samples.")

    result = []
    for entry in tqdm(entries, desc="PRV eval"):
        question = entry.get("question", "")
        if "asqa" in args.input_jsonl.lower():
            ctxs = entry.get("docs", entry.get("ctxs", []))
        else:
            ctxs = entry.get("ctxs", [])
        if not question:
            entry["response"] = [""]
            entry["llm_stats"] = {"total_llm_calls": 0, "iterations": 0, "call_breakdown": {}}
            entry["efficiency_stats"] = create_efficiency_stats()
            result.append(entry)
            continue
        tracer = Tracer()
        tracer_context.set(tracer)
        stats = create_efficiency_stats()
        token = efficiency_stats_context.set(stats)
        if getattr(args, "track_vram", False):
            try:
                import torch
                if getattr(torch, "cuda", None) and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        t_start = time.perf_counter()
        try:
            if zero_shot:
                answer = run_prv_eval_zeroshot(
                    question,
                    is_fever=is_fever_dataset,
                    is_eli5=is_eli5_dataset,
                    is_short_phrase=is_short_phrase_dataset,
                )
            elif use_planning:
                answer = run_prv_eval_with_planning(
                    question,
                    ctxs,
                    eval_top_k=args.eval_top_k,
                    recovery_top_k=getattr(args, "recovery_top_k", None),
                    verbose=args.verbose,
                    is_fever=is_fever_dataset,
                    is_eli5=is_eli5_dataset,
                    is_short_phrase=is_short_phrase_dataset,
                    no_recovery=args.no_recovery,
                    planning_only=args.planning_only,
                    use_improved_recovery=(getattr(args, "recovery_mode", "origin") == "improved"),
                )
            else:
                answer = run_prv_eval_single(
                    question,
                    ctxs,
                    max_docs=args.max_docs,
                    verbose=args.verbose,
                    is_fever=is_fever_dataset,
                    is_eli5=is_eli5_dataset,
                    is_short_phrase=is_short_phrase_dataset,
                )
            tracer.commit_pending()
            call_breakdown = {}
            latency_per_call_s = []
            output_lengths = []
            total_input_chars = 0
            for log_entry in tracer.log:
                name = log_entry.get("adapter_function_name", "unknown")
                call_breakdown[name] = call_breakdown.get(name, 0) + 1
                if "duration_s" in log_entry:
                    latency_per_call_s.append(log_entry["duration_s"])
                out_str = log_entry.get("llm_output") or ""
                output_lengths.append(len(str(out_str)))
                for _k, v in (log_entry.get("llm_inputs") or {}).items():
                    total_input_chars += len(str(v))
            total_output_chars = sum(output_lengths)
            # 粗略 token 估算（chars/4）
            est_out_tokens = total_output_chars // 4
            est_in_tokens = total_input_chars // 4
            iterations = call_breakdown.get("update_plan", 0) + call_breakdown.get("replan_conditions", 0)
            entry["latency_s"] = time.perf_counter() - t_start
            entry["llm_stats"] = {
                "total_llm_calls": len(tracer.log),
                "iterations": iterations,
                "call_breakdown": call_breakdown,
                "output_lengths": output_lengths,
                "total_output_chars": total_output_chars,
                "total_input_chars": total_input_chars,
                "estimated_output_tokens": est_out_tokens,
                "estimated_input_tokens": est_in_tokens,
            }
            if latency_per_call_s:
                entry["llm_stats"]["latency_per_call_s"] = latency_per_call_s
        except Exception as e:
            tracer.commit_pending()
            answer = f"Error: {e}"
            entry["latency_s"] = time.perf_counter() - t_start
            entry["llm_stats"] = {"total_llm_calls": len(tracer.log), "iterations": 0, "call_breakdown": {}}
        finally:
            if getattr(args, "track_vram", False):
                try:
                    import torch
                    if getattr(torch, "cuda", None) and torch.cuda.is_available():
                        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
                        stats["peak_vram_mb"] = peak_mb
                        entry["peak_vram_mb"] = peak_mb
                except Exception:
                    entry["peak_vram_mb"] = None
            else:
                entry["peak_vram_mb"] = None
            efficiency_stats_context.reset(token)
        entry["efficiency_stats"] = stats
        entry["response"] = [answer]
        # 单条 EM（用于效率画像：fail-safe 触发样本的准确率）
        golds = entry.get("answers", entry.get("answer", []))
        if not isinstance(golds, list):
            golds = [golds] if golds else []
        pred = answer.strip() if answer else ""
        entry["correct"] = any(_exact_match_score(pred, g) for g in golds) if golds else None
        result.append(entry)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(result)} items to {args.output_json}")

    # ---------- 效率画像：多维度汇总 ----------
    agg = {
        "parse_index_fail": 0, "parse_index_total": 0, "json_fail": 0, "json_total": 0,
        "failsafe_samples": 0, "total_samples": 0,
        "bridge_retention_checks": 0, "bridge_retention_ok": 0,
        "recovery_attempts": 0, "recovery_successes": 0,
        "recovery_triggered_samples": 0, "correct_recovery_triggered": 0,
        "total_facts": 0, "failed_facts": 0, "partial_facts": 0,
        "total_wall_clock_s": 0.0, "total_estimated_input_tokens": 0, "total_estimated_output_tokens": 0,
        "all_output_lengths": [],
        "peak_vram_mb_list": [],
    }
    correct_all = 0
    correct_failsafe = 0
    count_failsafe = 0
    correct_no_failsafe = 0
    count_no_failsafe = 0
    for r in result:
        es = r.get("efficiency_stats") or {}
        agg["parse_index_fail"] += es.get("parse_index_fail", 0)
        agg["parse_index_total"] += es.get("parse_index_total", 0)
        agg["json_fail"] += es.get("json_fail", 0)
        agg["json_total"] += es.get("json_total", 0)
        agg["bridge_retention_checks"] += es.get("bridge_retention_checks", 0)
        agg["bridge_retention_ok"] += es.get("bridge_retention_ok", 0)
        agg["recovery_attempts"] += es.get("recovery_attempts", 0)
        agg["recovery_successes"] += es.get("recovery_successes", 0)
        if es.get("recovery_attempts", 0) > 0:
            agg["recovery_triggered_samples"] += 1
            if r.get("correct") is True:
                agg["correct_recovery_triggered"] += 1
        agg["total_facts"] += es.get("total_facts", 0)
        agg["failed_facts"] += es.get("failed_facts", 0)
        agg["partial_facts"] += es.get("partial_facts", 0)
        agg["total_wall_clock_s"] += r.get("latency_s") or 0
        ls = r.get("llm_stats") or {}
        agg["total_estimated_input_tokens"] += ls.get("estimated_input_tokens", 0)
        agg["total_estimated_output_tokens"] += ls.get("estimated_output_tokens", 0)
        agg["all_output_lengths"].extend(ls.get("output_lengths") or [])
        if es.get("failsafe_triggered"):
            agg["failsafe_samples"] += 1
            if r.get("correct") is True:
                correct_failsafe += 1
            count_failsafe += 1
        else:
            if r.get("correct") is True:
                correct_no_failsafe += 1
            count_no_failsafe += 1
        agg["total_samples"] += 1
        if r.get("correct") is True:
            correct_all += 1
        pv = r.get("peak_vram_mb")
        if pv is not None:
            agg["peak_vram_mb_list"].append(pv)

    n = agg["total_samples"]
    parse_rate = (agg["parse_index_fail"] / agg["parse_index_total"] * 100) if agg["parse_index_total"] else 0
    json_rate = (agg["json_fail"] / agg["json_total"] * 100) if agg["json_total"] else 0
    em_all = (correct_all / n * 100) if n else 0
    em_failsafe = (correct_failsafe / count_failsafe * 100) if count_failsafe else None
    em_no_failsafe = (correct_no_failsafe / count_no_failsafe * 100) if count_no_failsafe else None
    bridge_rate = (agg["bridge_retention_ok"] / agg["bridge_retention_checks"] * 100) if agg["bridge_retention_checks"] else None
    recovery_rate = (agg["recovery_successes"] / agg["recovery_attempts"] * 100) if agg["recovery_attempts"] else None
    n_recovery_triggered = agg["recovery_triggered_samples"]
    em_recovery_triggered = (agg["correct_recovery_triggered"] / n_recovery_triggered * 100) if n_recovery_triggered else None
    false_fact_rate = (agg["failed_facts"] / agg["total_facts"] * 100) if agg["total_facts"] else None
    partial_fact_rate = (agg["partial_facts"] / agg["total_facts"] * 100) if agg["total_facts"] else None
    total_wall = agg["total_wall_clock_s"]
    qps = (n / total_wall) if total_wall and n else None
    ol = agg["all_output_lengths"]
    out_len_mean = (sum(ol) / len(ol)) if ol else None
    out_len_p50 = sorted(ol)[len(ol) // 2] if ol else None
    out_len_p95 = sorted(ol)[min(len(ol) - 1, int(len(ol) * 0.95))] if ol else None

    peak_vram_max = max(agg["peak_vram_mb_list"]) if agg["peak_vram_mb_list"] else None
    efficiency_profile = {
        "recovery_mode": getattr(args, "recovery_mode", "origin"),
        "recovery_top_k": getattr(args, "recovery_top_k", None),
        "peak_vram_mb": peak_vram_max,
        "note_peak_vram": "Client-side (e.g. E5 encoder). For vLLM server peak VRAM use nvidia-smi or run_vram_comparison." if peak_vram_max is not None else None,
        "parse_index_fail_rate_pct": round(parse_rate, 2),
        "parse_index_fail_count": agg["parse_index_fail"],
        "parse_index_total": agg["parse_index_total"],
        "json_parse_fail_rate_pct": round(json_rate, 2),
        "json_fail_count": agg["json_fail"],
        "json_total": agg["json_total"],
        "failsafe_triggered_samples": agg["failsafe_samples"],
        "total_samples": n,
        "failsafe_trigger_rate_pct": round(agg["failsafe_samples"] / n * 100, 2) if n else 0,
        "em_overall_pct": round(em_all, 2),
        "em_when_failsafe_triggered_pct": round(em_failsafe, 2) if em_failsafe is not None else None,
        "em_when_no_failsafe_pct": round(em_no_failsafe, 2) if em_no_failsafe is not None else None,
        "bridge_evidence_retention_rate_pct": round(bridge_rate, 2) if bridge_rate is not None else None,
        "bridge_retention_checks": agg["bridge_retention_checks"],
        "bridge_retention_ok": agg["bridge_retention_ok"],
        "recovery_success_rate_pct": round(recovery_rate, 2) if recovery_rate is not None else None,
        "recovery_attempts": agg["recovery_attempts"],
        "recovery_successes": agg["recovery_successes"],
        "recovery_triggered_samples": n_recovery_triggered,
        "em_recovery_triggered_pct": round(em_recovery_triggered, 2) if em_recovery_triggered is not None else None,
        "correct_recovery_triggered": agg["correct_recovery_triggered"],
        "false_fact_rate_pct": round(false_fact_rate, 2) if false_fact_rate is not None else None,
        "partial_fact_rate_pct": round(partial_fact_rate, 2) if partial_fact_rate is not None else None,
        "total_facts": agg["total_facts"],
        "failed_facts": agg["failed_facts"],
        "partial_facts": agg["partial_facts"],
        "total_wall_clock_s": round(total_wall, 2),
        "qps": round(qps, 4) if qps is not None else None,
        "total_estimated_input_tokens": agg["total_estimated_input_tokens"],
        "total_estimated_output_tokens": agg["total_estimated_output_tokens"],
        "output_length_chars_mean": round(out_len_mean, 1) if out_len_mean is not None else None,
        "output_length_chars_p50": out_len_p50,
        "output_length_chars_p95": out_len_p95,
    }
    profile_path = args.output_json.rsplit(".", 1)[0] + "_efficiency_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(efficiency_profile, f, ensure_ascii=False, indent=2)
    print("\n--- 效率画像 (Efficiency Profile) ---")
    print("  [Fail-safe / 解析鲁棒性]")
    print(f"    ParseIndex 失败触发率: {efficiency_profile['parse_index_fail_rate_pct']}%  ({agg['parse_index_fail']}/{agg['parse_index_total']})")
    print(f"    JSON 解析失败触发率:  {efficiency_profile['json_parse_fail_rate_pct']}%  ({agg['json_fail']}/{agg['json_total']})")
    print(f"    Fallback 触发率:       {efficiency_profile['failsafe_trigger_rate_pct']}%  ({agg['failsafe_samples']}/{n} 样本)")
    print("  [闭环稳定性]")
    if bridge_rate is not None:
        print(f"    Bridge evidence 保留率: {efficiency_profile['bridge_evidence_retention_rate_pct']}%  ({agg['bridge_retention_ok']}/{agg['bridge_retention_checks']})")
    if recovery_rate is not None:
        print(f"    Recovery 成功率:        {efficiency_profile['recovery_success_rate_pct']}%  ({agg['recovery_successes']}/{agg['recovery_attempts']})")
    if em_recovery_triggered is not None:
        print(f"    样本级 Recovery EM:      {efficiency_profile['em_recovery_triggered_pct']}%  ({agg['correct_recovery_triggered']}/{n_recovery_triggered} 至少触发一次 recovery 的样本)")
    if false_fact_rate is not None:
        print(f"    False/未证实事实率:     {efficiency_profile['false_fact_rate_pct']}% (FAILED)  {efficiency_profile.get('partial_fact_rate_pct')}% (PARTIAL)")
    print("  [性能 / Token / Latency / 显存]")
    if peak_vram_max is not None:
        print(f"    Peak VRAM (client): {peak_vram_max} MB  (vLLM 服务端显存需另测)")
    print(f"    Wall-clock: {efficiency_profile['total_wall_clock_s']} s   QPS: {efficiency_profile.get('qps')}")
    print(f"    Est. tokens: input={efficiency_profile['total_estimated_input_tokens']}  output={efficiency_profile['total_estimated_output_tokens']}")
    if out_len_mean is not None:
        print(f"    输出长度(chars): mean={efficiency_profile.get('output_length_chars_mean')}  p50={efficiency_profile.get('output_length_chars_p50')}  p95={efficiency_profile.get('output_length_chars_p95')}")
    print("  [EM]")
    print(f"    总体: {efficiency_profile['em_overall_pct']}%  |  触发 fail-safe 样本: {efficiency_profile.get('em_when_failsafe_triggered_pct')}%  |  未触发: {efficiency_profile.get('em_when_no_failsafe_pct')}%")
    print(f"  Profile saved to {profile_path}")


if __name__ == "__main__":
    main()
