"""
PRV 评测脚本：使用评测集自带的 (question, ctxs)，与 DynamicRAG 同输入同评估。

- 默认（--use-planning）：完整 PRV = 规划(拆子问题) + 多轮(每轮从 ctxs 做 E5 排序→重排→事实提取→更新计划) → 合成。
  检索仅限当前题 ctxs，用 E5 在 ctxs 上做相似度排序再取 top-k，避免按位置截断把高相关 doc 截掉。
- --no-planning：单轮 ctxs→重排→事实提取→合成（兼容旧行为）。
"""
import argparse
import json
import re
import sys
import os

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
# 同目录下的评测检索模块（E5 仅在 ctxs 上排序）
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import eval_retrieve


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


def run_prv_eval_single(question: str, ctxs: list, max_docs: int = 40, verbose: bool = False) -> str:
    """
    单条评测（无规划）：给定 question + ctxs，ctxs 按位置截断 → 重排 → 事实提取 → 答案合成。
    """
    search_hits = ctxs_to_search_hits(ctxs, max_docs=max_docs)
    if not search_hits:
        return "No relevant context provided."

    reranked = prv_reranker.rerank_documents(question, search_hits)
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
        collected_facts = {"reasoned_facts": []}
    else:
        try:
            extracted = json.loads(match.group(0))
            if "reasoned_facts" not in extracted and isinstance(extracted, dict) and "statement" in extracted:
                collected_facts = {"reasoned_facts": [extracted]}
            else:
                collected_facts = extracted if isinstance(extracted, dict) and "reasoned_facts" in extracted else {"reasoned_facts": []}
        except json.JSONDecodeError:
            collected_facts = {"reasoned_facts": []}

    answer = rag_core.synthesize_final_answer(question, collected_facts)
    return answer.strip() if answer else ""


def run_prv_eval_with_planning(
    question: str,
    ctxs: list,
    eval_top_k: int = 40,
    max_iterations: int = 5,
    max_total_attempts: int = 10,
    verbose: bool = False,
) -> str:
    """
    完整 PRV 评测：规划(拆子问题) + 多轮迭代，每轮「从 ctxs 用 E5 排序取 top-k → 重排 → 事实提取 → 更新计划」→ 合成。
    检索仅限当前题 ctxs，不查外部 E5 大库。
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
        # 退化为单需求
        all_requirements = [{"requirement_id": "req1", "question": question, "depends_on": None}]
    else:
        all_requirements = analysis_result["requirements"]

    pending_requirements = list(all_requirements)
    req_id_to_question = {req["requirement_id"]: req["question"] for req in all_requirements}
    collected_facts = {"reasoned_facts": []}
    last_extraction_was_direct_only = True
    iteration_count = 0
    total_attempt_count = 0

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
            # SP：update_plan 或 replan_questions
            if last_extraction_was_direct_only:
                decision_result = rag_core.update_plan(
                    query=question, collected_facts=collected_facts, pending_requirements=pending_requirements
                )
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

            next_actions = decision.get("next_actions") or decision.get("next_questions", [])
            if not next_actions:
                raise ValueError("Planner said CONTINUE_SEARCH but gave no next_actions.")

            iteration_new_facts = []
            for action in next_actions:
                if not any(req["requirement_id"] == action.get("requirement_id") for req in pending_requirements):
                    continue
                req = [r for r in pending_requirements if r["requirement_id"] == action.get("requirement_id")][0]
                sub_query = action.get("question") or req["question"]
                # 从 ctxs 用 E5 做相似度排序取 top-k，再重排+事实提取
                search_hits = eval_retrieve.rank_ctxs_by_query(sub_query, ctxs, k=eval_top_k)
                extracted = rag_core.extract_facts_given_hits(
                    search_query=sub_query,
                    requirement=req,
                    collected_facts=collected_facts,
                    search_hits=search_hits,
                )
                if isinstance(extracted, dict) and extracted.get("reasoned_facts"):
                    iteration_new_facts.extend(extracted["reasoned_facts"])

            if iteration_new_facts:
                processed = []
                for fact in iteration_new_facts:
                    req_id = fact.get("fulfills_requirement_id")
                    if not req_id:
                        continue
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

            iteration_count += 1
        except (json.JSONDecodeError, ValueError, RuntimeError) as e:
            if verbose:
                print(f"  iteration failed: {e}, rolling back")
            collected_facts["reasoned_facts"] = facts_before
            pending_requirements = pending_before
            req_id_to_question = req_map_before
            last_extraction_was_direct_only = last_direct_before
            continue

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
    parser.add_argument("--sample-size", type=int, default=None, help="Only run first N samples (for quick test)")
    parser.add_argument("--verbose", action="store_true", help="Print per-item progress")
    args = parser.parse_args()
    use_planning = args.use_planning and not args.no_planning

    print("Configuring LLM provider...")
    llm_adapter.configure_llm_provider()
    if use_planning:
        print("Eval mode: full PRV (planning + E5-on-ctxs retrieval). Pre-loading E5 encoder...")
        eval_retrieve.get_e5_encoder()
    print("Loading input...")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if args.sample_size is not None and args.sample_size > 0:
        entries = entries[: args.sample_size]
        print(f"Quick test: using first {len(entries)} samples only.")

    result = []
    for entry in tqdm(entries, desc="PRV eval"):
        question = entry.get("question", "")
        if "asqa" in args.input_jsonl.lower():
            ctxs = entry.get("docs", entry.get("ctxs", []))
        else:
            ctxs = entry.get("ctxs", [])
        if not question:
            entry["response"] = [""]
            result.append(entry)
            continue
        try:
            if use_planning:
                answer = run_prv_eval_with_planning(
                    question, ctxs, eval_top_k=args.eval_top_k, verbose=args.verbose
                )
            else:
                answer = run_prv_eval_single(question, ctxs, max_docs=args.max_docs, verbose=args.verbose)
        except Exception as e:
            answer = f"Error: {e}"
        entry["response"] = [answer]
        result.append(entry)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(result)} items to {args.output_json}")


if __name__ == "__main__":
    main()
