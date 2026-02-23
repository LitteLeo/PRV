"""
REAP框架主流程模块（pipeline.py）

本模块实现了REAP（Recursive Evaluation and Adaptive Planning）框架的完整执行流程，
通过"分解-迭代规划-事实提取-合成"的闭环流程，实现多跳问答（MHQA）的精准推理。

REAP框架完整流程：
1. 阶段1：初始查询分解（Decomposer模块）
   - 将复杂查询Q拆解为结构化初始任务计划P₀
2. 阶段2：核心迭代循环（SP与FE协同）
   - 子步骤2.1：SP分析状态，确定可执行动作Actionsₜ
   - 子步骤2.2：FE处理Actionsₜ，提取结构化事实fₜ
   - 子步骤2.3：SP更新计划与事实，进入下一轮迭代
   - 迭代终止条件：所有子任务完成、连续失败、达到最大迭代次数
3. 阶段3：答案合成（Synthesizer模块）
   - 基于最终事实列表F_final合成最终答案A

迭代终止条件（满足任一即停止）：
1. 所有子任务完成，事实列表Fₜ已覆盖原始查询所需信息
2. 连续多轮（如2轮）提取到PartialClue/Failed，且Re-Planner无法生成有效新子任务
3. 迭代次数达到预设上限（默认5次）
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_pipeline_lib import core as rag_core
from contextvars import ContextVar, copy_context
from functools import partial
import json
import copy

# 追踪器上下文变量，用于记录LLM调用和迭代信息
tracer_context: ContextVar = ContextVar('tracer_context', default=None)

# 效率画像：使用 rag_pipeline_lib.efficiency_stats 的 context，由评测脚本在每样本设置
from rag_pipeline_lib.efficiency_stats import efficiency_stats_context

class Tracer:
    """
    追踪器类：用于记录REAP框架执行过程中的LLM调用和迭代信息
    
    功能：
    - 记录每次LLM调用的输入和输出
    - 跟踪迭代次数
    - 支持提交和丢弃待处理的日志（用于错误回滚）
    """
    def __init__(self):
        self.log = []  # 已提交的日志列表
        self.iteration_count = 0  # 当前迭代次数
        self._pending_log = []  # 待处理的日志列表（用于错误回滚）

    def record_llm_call(self, adapter_function_name, inputs, output, duration_s=None):
        """记录一次LLM调用，可选记录单次耗时 duration_s（秒）。"""
        trace_entry = {
            "adapter_function_name": adapter_function_name,
            "llm_inputs": copy.deepcopy(inputs),
            "llm_output": copy.deepcopy(output)
        }
        if duration_s is not None:
            trace_entry["duration_s"] = duration_s
        self._pending_log.append(trace_entry)

    def commit_pending(self):
        """提交待处理的日志到正式日志列表"""
        self.log.extend(self._pending_log)
        self._pending_log.clear()

    def discard_pending(self):
        """丢弃待处理的日志（用于错误回滚）"""
        self._pending_log.clear()

def run_multistep_pipeline(query: str, verbose: bool = True, trace_collector: Tracer = None, serial_next_actions: bool = False) -> str:
    """
    REAP框架主执行函数：实现完整的"分解-迭代规划-事实提取-合成"流程
    
    这是REAP框架的入口函数，协调Decomposer、SP（子任务规划器）和FE（事实提取器）
    三个核心模块，通过递归协同实现多跳问答的精准推理。
    
    流程概述：
    1. 阶段1：初始查询分解 - 调用analyze_and_decompose_query生成初始任务计划P₀
    2. 阶段2：核心迭代循环 - SP与FE协同，逐步完善事实列表和任务计划
    3. 阶段3：答案合成 - 调用synthesize_final_answer生成最终答案
    
    Args:
        query: 用户的复杂多跳查询Q
        verbose: 是否打印详细执行信息
        trace_collector: 可选的追踪器对象，用于记录执行过程
        serial_next_actions: 若 True，next_actions 串行执行（用于 Parallel vs Serial 对比实验、wall-clock/QPS）
        
    Returns:
        str: 原始查询的最终答案A
    """
    current_tracer = tracer_context.get()

    if verbose:
        print(f"\n🚀 Starting new multi-step pipeline for query: '{query}'")

    # ========== 阶段1：初始查询分解（Decomposer模块）==========
    # 功能：将复杂查询Q拆解为结构化初始任务计划P₀ = {p₁, p₂, ..., pₙ}
    # 每个子任务pᵢ的格式为(idᵢ, qᵢ, depsᵢ)
    analysis_result = None
    max_analysis_retries = 3  # 最大重试次数
    # 重试机制：如果查询分解失败，最多重试3次
    for attempt in range(max_analysis_retries):
        try:
            if verbose and attempt > 0:
                print(f"🔄 Retrying query analysis... (Attempt {attempt + 1}/{max_analysis_retries})")
            
            # 丢弃之前失败的日志（如果有）
            if current_tracer: current_tracer.discard_pending()
            # 调用Decomposer模块进行查询分解
            analysis_result = rag_core.analyze_and_decompose_query(query=query)
            
            # 验证分解结果的有效性
            if not analysis_result or "requirements" not in analysis_result:
                raise ValueError("Analysis result is empty or missing 'requirements' key.")
            
            # 提交成功的日志
            if current_tracer: current_tracer.commit_pending()
            if verbose: print("✅ Query analysis successful.")
            break 
        except Exception as e:
            if verbose:
                print(f"❌ Query analysis failed on attempt {attempt + 1}. Error: {e}")
            # 如果达到最大重试次数，返回错误
            if attempt == max_analysis_retries - 1:
                if current_tracer: current_tracer.discard_pending()
                error_msg = f"Pipeline Error: Failed to analyze and decompose the query after {max_analysis_retries} attempts."
                if verbose: print(f"\n❌ {error_msg}")
                return error_msg
    
    if not analysis_result:
        return "Pipeline Error: Could not obtain a valid query analysis."

    if verbose:
        print("--- Initial Analysis and Requirements ---")
        print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
        print("------------------------------------")

    # ========== 初始化状态变量 ==========
    # 这些变量将在迭代循环中不断更新
    all_requirements = analysis_result.get("requirements", [])  # 所有子任务列表（初始任务计划P₀）
    pending_requirements = list(all_requirements)  # 待完成的子任务列表（当前任务计划P_t）
    req_id_to_question = {req['requirement_id']: req['question'] for req in all_requirements}  # 子任务ID到问题的映射
    collected_facts = {"reasoned_facts": []}  # 收集的事实列表F_t（初始为空F₀=∅）
    max_iterations = 5  # 最大迭代次数（论文中设为5）
    max_total_attempts = 10  # 最大总尝试次数（包括失败重试）
    last_extraction_was_direct_only = True  # 标记上一轮提取是否全部为DirectAnswer（用于选择Plan Updater或Re-Planner） 

    # ========== 阶段2：核心迭代循环（SP与FE协同）==========
    # 这是REAP框架的核心环节，通过SP（战略规划）与FE（事实采集）的递归交互，
    # 逐步完善事实列表、优化任务计划，直到满足终止条件
    iteration_count = 0  # 当前迭代次数（成功完成的迭代数）
    total_attempt_count = 0  # 总尝试次数（包括失败重试）
    while iteration_count < max_iterations:
        total_attempt_count += 1  # 每次循环增加总尝试次数
        # 终止条件1：达到最大总尝试次数（防止无限重试）
        if total_attempt_count > max_total_attempts:
            if verbose: print(f"\n⚠️ Warning: Reached maximum total attempts ({max_total_attempts}) due to repeated failures. Moving to synthesis.")
            break

        # 终止条件2：所有子任务已完成（理想终止条件）
        if not pending_requirements:
            if verbose: print("\n✅ All requirements have been fulfilled. Moving to final answer synthesis.")
            break
        
        # 更新追踪器的迭代计数
        if current_tracer:
            current_tracer.iteration_count = iteration_count + 1
            # 在每次迭代开始时，丢弃之前失败的日志（用于错误回滚）
            current_tracer.discard_pending()

        if verbose:
            print(f"\n--- Iteration {iteration_count + 1}/{max_iterations} ---")
            print(f"📝 Pending Requirements: {[req['question'] for req in pending_requirements]}")

        # ========== 状态快照：保存迭代开始前的状态（用于错误回滚）==========
        # 如果迭代过程中发生错误，可以回滚到这些状态
        facts_before_iteration = [fact for fact in collected_facts["reasoned_facts"]]  # 迭代前的事实列表F_{t-1}
        pending_reqs_before_iteration = list(pending_requirements)  # 迭代前的待完成计划P_{t-1}
        req_map_before_iteration = dict(req_id_to_question)  # 迭代前的ID映射
        last_direct_before_iteration = last_extraction_was_direct_only  # 迭代前的提取状态标记
        
        try:
            # ========== 子步骤2.1：SP分析状态，确定可执行动作Actionsₜ ==========
            # SP从全局视角评估当前推理状态，判断哪些子任务已满足依赖条件（即前置子任务的事实已提取），
            # 将其确定为"可执行动作"（Actionsₜ）
            # 
            # 根据上一轮提取结果的满足度标签lₜ，选择不同的规划策略：
            # - 若lₜ=DirectAnswer（理想场景）→ 调用Plan Updater（轻量级，执行事实替换和计划分叉）
            # - 若lₜ=PartialClue/Failed（非理想场景）→ 调用Re-Planner（完整规划，执行实用充分性评估和范围化计划修复）
            decision_result = None
            if last_extraction_was_direct_only:
                # 上一轮提取全部为DirectAnswer，使用轻量级Plan Updater
                if verbose: print("\n🔄 Last extraction was successful. Using lightweight 'update_plan'.")
                decision_result = rag_core.update_plan(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)
            else:
                # 上一轮提取包含PartialClue或Failed，使用完整Re-Planner
                if verbose: print("\n🤔 Last extraction had partial clues or failures. Using full 'replan_questions'.")
                decision_result = rag_core.replan_questions(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)

            # 验证规划结果的有效性
            if not decision_result or "decision" not in decision_result:
                raise ValueError("Planning step failed to return a valid decision.")
            if verbose:
                print("--- Planning Decision ---")
                print(json.dumps(decision_result, indent=2, ensure_ascii=False))
                print("-------------------------")
            decision = decision_result.get("decision", {})

            # ========== 子步骤2.3（部分）：SP更新任务计划P_t ==========
            # 根据SP返回的updated_plan，更新待完成的子任务列表
            # 这对应论文中的计划更新：P_t = SP(P_{t-1}, F_t, Q)
            if "updated_plan" in decision and isinstance(decision["updated_plan"], list):
                if verbose: print("🔄 Updating pending requirements based on the new plan.")
                pending_requirements = decision["updated_plan"]  # 更新为新的任务计划P_t
                req_id_to_question = {req['requirement_id']: req['question'] for req in pending_requirements}  # 更新ID映射

            # 检查SP的决策：是否继续搜索或直接合成答案
            next_step = decision.get("next_step")
            if next_step == "SYNTHESIZE_ANSWER":
                # SP判断所有必要事实已收集，可以进入答案合成阶段
                if verbose: print("✅ Planning module decided all necessary facts are collected. Moving to synthesis.")
                if current_tracer: current_tracer.commit_pending()
                break
            
            # 验证next_step的有效性
            if next_step != "CONTINUE_SEARCH":
                raise ValueError(f"Received unexpected next step '{next_step}'.")

            # 获取下一轮可执行动作列表Actions_{t+1}
            # 这些动作是SP根据依赖关系判断出的、可以立即执行的子任务
            next_actions = decision.get("next_actions") or decision.get("next_questions", [])
            if not next_actions:
                raise ValueError("Planner suggested to continue search, but provided no actions.")

            # ========== 子步骤2.2：FE处理Actionsₜ，提取结构化事实fₜ ==========
            # 为Actionsₜ中的每个子任务pᵢ，通过"检索→分析→提取"三步，生成高保真的结构化事实
            # 对应论文公式：f_t = M_θ(ExtractF | q_t, D_t, F_{t-1}) （公式7）
            # 
            # 并行或串行执行 next_actions（serial_next_actions=True 用于 Parallel vs Serial 对比）
            if verbose: print(f"\n🔎 Executing {len(next_actions)} search action(s) {'serially' if serial_next_actions else 'in parallel'}...")
            iteration_new_facts = []  # 本轮迭代提取的新事实列表{f₁, f₂, ..., fₖ}
            extraction_had_errors = False  # 标记是否有提取错误

            actions_to_run = [a for a in next_actions if any(req['requirement_id'] == a.get("requirement_id") for req in pending_requirements)]

            if serial_next_actions:
                for action in actions_to_run:
                    req = [req for req in pending_requirements if req['requirement_id'] == action.get("requirement_id")][0]
                    newly_extracted_data = rag_core.retrieve_and_extract_facts(
                        search_query=action.get("question"),
                        requirement=req,
                        collected_facts=collected_facts,
                    )
                    if not isinstance(newly_extracted_data, dict) or "reasoned_facts" not in newly_extracted_data:
                        raise ValueError(f"Invalid data structure received for '{action.get('question')}'.")
                    iteration_new_facts.extend(newly_extracted_data.get("reasoned_facts", []))
                    if verbose:
                        print(f"  - 📝 Result for '{action.get('question')}': {len(newly_extracted_data.get('reasoned_facts', []))} fact(s)")
            else:
                # 使用线程池并行执行多个子任务的事实提取
                with ThreadPoolExecutor() as executor:
                    future_to_action = {
                        executor.submit(
                            partial(copy_context().run, rag_core.retrieve_and_extract_facts),
                            search_query=action.get("question"),
                            requirement=[req for req in pending_requirements if req['requirement_id'] == action.get("requirement_id")][0],
                            collected_facts=collected_facts
                        ): action
                        for action in actions_to_run
                    }
                    for future in as_completed(future_to_action):
                        action = future_to_action[future]
                        try:
                            newly_extracted_data = future.result()
                            if verbose:
                                print(f"  - 📝 Result for '{action.get('question')}':")
                                try:
                                    print(f"    {json.dumps(newly_extracted_data, indent=4, ensure_ascii=False)}")
                                except (TypeError, json.JSONDecodeError):
                                    print(f"    (Could not format non-JSON or invalid JSON output: {newly_extracted_data})")
                            if not isinstance(newly_extracted_data, dict) or "reasoned_facts" not in newly_extracted_data:
                                raise ValueError(f"Invalid data structure received for '{action.get('question')}'.")
                            iteration_new_facts.extend(newly_extracted_data.get("reasoned_facts", []))
                        except Exception as exc:
                            if verbose: print(f"  - ❌ Error processing result for '{action.get('question')}': {exc}")
                            raise RuntimeError(f"Fact extraction failed for '{action.get('question')}'") from exc

            if verbose:
                print(f"  - ✅ Fact extraction phase completed. Found {len(iteration_new_facts)} new fact(s).")

            # ========== 子步骤2.3（部分）：SP更新事实列表F_t ==========
            # 将新提取的事实合并到历史事实中，对应论文公式：F_t = F_{t-1} ∪ {f₁, f₂, ..., fₖ}
            # 同时根据满足度标签lₜ，更新待完成子任务列表和提取状态标记
            if iteration_new_facts:
                # 处理提取的事实，确保格式统一
                processed_facts = []
                for fact in iteration_new_facts:
                    req_id = fact.get("fulfills_requirement_id")
                    if not req_id:
                        continue

                    question = req_id_to_question.get(req_id, "Unknown Question")
                    
                    # 构建处理后的结构化事实（对应公式8：f_t = (s_t, e_t, r_t, l_t)）
                    processed_fact = {
                        "fulfills_requirement_id": req_id,  # 满足的子任务ID
                        "requirement": question,  # 子任务问题
                        "reasoning": fact.get("reasoning"),  # 推理过程r_t
                        "statement": fact.get("statement"),  # 核心陈述s_t
                        "fulfillment_level": fact.get("fulfillment_level")  # 满足度标签l_t
                    }
                    processed_fact = {k: v for k, v in processed_fact.items() if v is not None}

                    processed_facts.append(processed_fact)

                # 更新事实列表：F_t = F_{t-1} ∪ {f₁, f₂, ..., fₖ}
                collected_facts["reasoned_facts"].extend(processed_facts)
                
                # 根据满足度标签lₜ，移除已完成（DIRECT_ANSWER）的子任务
                fulfilled_req_ids = {fact['fulfills_requirement_id'] for fact in iteration_new_facts if fact.get("fulfillment_level") == "DIRECT_ANSWER"}
                pending_requirements = [req for req in pending_requirements if req['requirement_id'] not in fulfilled_req_ids]
                
                # 更新提取状态标记：如果所有事实都是DIRECT_ANSWER且无错误，标记为True（下次使用Plan Updater）
                last_extraction_was_direct_only = all(fact.get("fulfillment_level") == "DIRECT_ANSWER" for fact in iteration_new_facts) and not extraction_had_errors
            else:
                # 如果没有提取到新事实，标记为False（下次使用Re-Planner）
                last_extraction_was_direct_only = False

            # ========== 迭代成功完成 ==========
            # 提交日志并增加迭代计数
            if current_tracer: current_tracer.commit_pending()
            iteration_count += 1

        except (json.JSONDecodeError, ValueError, RuntimeError) as e:
            # ========== 错误处理：状态回滚 ==========
            # 如果迭代过程中发生错误，回滚到迭代开始前的状态，然后继续下一轮尝试
            if verbose: 
                print(f"\n❌ Iteration {iteration_count + 1} failed: {e}")
                print("🔄 Rolling back state to the beginning of the iteration and retrying.")
            
            # 状态回滚：恢复到迭代开始前的状态
            collected_facts["reasoned_facts"] = facts_before_iteration  # 回滚事实列表
            pending_requirements = pending_reqs_before_iteration  # 回滚待完成计划
            req_id_to_question = req_map_before_iteration  # 回滚ID映射
            last_extraction_was_direct_only = last_direct_before_iteration  # 回滚提取状态标记
            
            # 丢弃失败的日志
            if current_tracer: current_tracer.discard_pending()
            
            # 继续下一轮尝试（不增加iteration_count，因为这次迭代失败了）
            continue
    else:
        # 终止条件3：达到最大迭代次数（非理想终止）
        if verbose: print("\n⚠️ Warning: Reached maximum iterations. Moving to synthesis with potentially incomplete facts.")

    # ========== 阶段3：答案合成（Synthesizer模块）==========
    # 功能：调用LLM，基于最终事实列表F_final中的所有事实，合成符合原始查询需求的最终答案
    # 对应论文公式：A = M_θ(Synthesize | Q, F_final) （公式4）
    if verbose:
        print("\n--- Final Stage: Synthesizing Answer from Collected Facts ---")
        print("Collected Facts Summary:")
        print(json.dumps(collected_facts, indent=2, ensure_ascii=False))
    
    # 丢弃待处理的日志（如果有）
    if current_tracer: current_tracer.discard_pending()
    # 调用Synthesizer模块生成最终答案
    final_answer = rag_core.synthesize_final_answer(query=query, collected_facts=collected_facts)
    # 提交最终答案生成的日志
    if current_tracer: current_tracer.commit_pending()
    return final_answer
