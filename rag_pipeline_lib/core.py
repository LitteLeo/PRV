"""
REAP框架核心模块（core.py）

本模块实现了REAP（Recursive Evaluation and Adaptive Planning）框架的核心功能，
包括查询分解、事实提取、计划更新和答案合成等关键步骤。

REAP框架流程概述：
1. 阶段1：初始查询分解（Decomposer模块）- analyze_and_decompose_query
2. 阶段2：核心迭代循环（SP与FE协同）
   - 子步骤2.1：SP分析状态，确定可执行动作 - update_plan/replan_questions
   - 子步骤2.2：FE处理Actions，提取结构化事实 - retrieve_and_extract_facts
   - 子步骤2.3：SP更新计划与事实 - update_plan/replan_questions
3. 阶段3：答案合成（Synthesizer模块）- synthesize_final_answer
"""
import config
from search import search_utils
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib import prv_reranker
from typing import List
import re
import json

def retrieve_context(query: str) -> list[dict]:
    """
    文档检索功能（FE模块的检索组件）
    
    功能：根据查询从外部检索服务获取相关文档，对应REAP论文中的公式1：
    D_t = Retriever(q_t; C)
    
    这是事实提取器（FE）的第一步，用于从语料库C中检索与子任务查询q_t相关的文档。
    论文中设置每次检索Top-5文档（由config.TOP_K控制），确保信息覆盖度。
    
    Args:
        query: 检索查询字符串，通常是子任务的查询q_t
        
    Returns:
        list[dict]: 检索到的文档列表，每个文档包含'contents'和可选的'title'字段
        如果检索失败，返回空列表[]
    """
    print(f"\n--- Stage: Retrieving context for query: '{query}' via HTTP search ---")
    try:
        # 调用检索服务（如e5-large-v2编码器）进行文档检索
        # 对应论文公式1：D_t = Retriever(q_t; C)
        search_hits = search_utils.search_by_http(
            query=query,
            k=config.TOP_K,  # Top-K文档数量（论文中设为5）
            host=config.SEARCH_SERVICE_HOST,
            port=config.SEARCH_SERVICE_PORT
        )
        return search_hits
    except Exception as e:
        print(f"Error during context retrieval via HTTP: {e}")
        return []

def perform_rag(retrieval_query: str, generation_query: str = None) -> str:
    actual_generation_query = generation_query if generation_query is not None else retrieval_query

    print(f"\n--- Performing Single-Stage RAG (with HTTP Search) ---")
    print(f"--- Retrieval Query: '{retrieval_query}' ---")
    if generation_query is not None and retrieval_query != generation_query:
        print(f"--- Generation Query: '{generation_query}' ---")

    # 1. Retrieve context using the new HTTP-based retrieval
    search_hits = retrieve_context(retrieval_query)

    # 2. Prepare context string
    context = "No relevant context found."
    if search_hits:
        retrieved_texts_with_tags = []
        for i, hit in enumerate(search_hits):
            context_content = ""
            title = ""
            if isinstance(hit, dict):
                if 'contents' in hit:
                    context_content = str(hit['contents'])
                else:
                    print(f"Warning: Search hit missing 'contents', using full hit: {str(hit)[:100]}...")
                    context_content = str(hit)
                
                if 'title' in hit:
                    title = str(hit['title'])
            else:
                context_content = str(hit)
            
            if title:
                retrieved_texts_with_tags.append(f'<document id={i+1} title="{title}">\n{context_content}\n</document>')
            else:
                retrieved_texts_with_tags.append(f'<document id={i+1}>\n{context_content}\n</document>')
        
        if retrieved_texts_with_tags:
            context = "\n\n".join(retrieved_texts_with_tags)

        print(f"--- Retrieved Context (for retrieval query: '{retrieval_query}') ---")
        if config.SHOW_RETRIEVED_CONTEXT:
            print(context)
        else:
            print("[Context display disabled]")
        print("---------------------------------------")
    else:
        print(f"No relevant context found for retrieval query: '{retrieval_query}'.")

    # 3. Generate response
    try:
        answer = llm_adapter.generate_rag_response(
            query=actual_generation_query,
            context=context
        )
    except Exception as e:
        print(f"Error generating response for generation query '{actual_generation_query}': {e}")
        answer = f"Error generating response for generation query '{actual_generation_query}'."

    print(f"\nGeneration Query: {actual_generation_query}")
    print(f"Generated Answer: {answer}")
    return answer

def analyze_and_decompose_query(query: str) -> dict:
    """
    阶段1：初始查询分解（Decomposer模块）
    
    功能：将用户的复杂多跳查询Q拆解为结构化初始任务计划P₀，明确子任务间的依赖关系。
    这是REAP框架的入口步骤，对应论文"阶段1：初始查询分解"。
    
    输入：用户的复杂多跳查询Q（如"歌曲《Week Without You》演唱者的生日是什么时候？"）
    输出：初始任务计划P₀ = {p₁, p₂, ..., pₙ}，每个子任务pᵢ的格式为(idᵢ, qᵢ, depsᵢ)
        - idᵢ：子任务唯一标识（如req1、req2）
        - qᵢ：子任务对应的具体查询（如req1："《Week Without You》的演唱者是谁？"）
        - depsᵢ：子任务依赖的前置子任务ID（如req2的depsᵢ=req1，需先完成req1才能执行req2）
    
    示例（文档1-28案例）：
        对问题"When is the performer of song 'Week Without You''s birthday?"，
        Decomposer生成P₀含两个子任务：
        - req1：Who is the performer of the song 'Week Without You'?（depsᵢ=null）
        - req2：What is the birthday of the performer identified in req1?（depsᵢ=req1）
    
    Args:
        query: 用户的输入查询Q

    Returns:
        dict: 包含以下键的字典：
            - "user_goal" (str): 用户目标的简要总结
            - "requirements" (list): 子任务列表，每个子任务包含：
                - "requirement_id" (str): 子任务ID（如"req1"）
                - "question" (str): 子任务的具体查询
                - "depends_on" (str|null): 依赖的前置子任务ID，无依赖则为null
        如果分析或解析失败，返回空字典{}
    """
    print(f"\n--- Stage: Analyzing and decomposing query: '{query}' ---")
    try:
        # 步骤1：调用LLM适配器进行查询分析和分解
        # LLM将根据SYSTEM_PROMPT_QUERY_ANALYSIS提示词，将复杂查询分解为原子子任务
        analysis_json_str = llm_adapter.analyze_query(query)

        # 步骤2：从LLM响应中提取JSON对象
        # LLM可能返回包含额外文本的响应，需要提取其中的JSON部分
        match = re.search(r'\{.*\}', analysis_json_str, re.DOTALL)
        if not match:
            print(f"Error: Could not find a JSON object in the LLM response.")
            print(f"Raw Response:\n{analysis_json_str}")
            return {}
        
        json_str = match.group(0)

        # 步骤3：将JSON字符串解析为Python字典
        # 解析后的字典应包含"user_goal"和"requirements"两个键
        analysis_result = json.loads(json_str)
        
        # 步骤4：验证解析结果的结构完整性
        # 确保包含必需的"user_goal"和"requirements"字段
        if "user_goal" not in analysis_result or "requirements" not in analysis_result:
            print("Error: Parsed JSON is missing 'user_goal' or 'requirements' key.")
            print(f"Parsed Data: {analysis_result}")
            return {}

        print("--- Query Analysis Successful ---")
        print(f"User Goal: {analysis_result.get('user_goal')}")
        print(f"Decomposed into {len(analysis_result.get('requirements', []))} sub-questions.")
        print("---------------------------------")
        
        # 返回初始任务计划P₀，后续将用于迭代循环
        return analysis_result

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from LLM response. {e}")
        print(f"Raw Response that caused error:\n{analysis_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during query analysis: {e}")
        return {}

def retrieve_and_extract_facts(search_query: str, requirement: dict, collected_facts: dict) -> dict:
    """
    阶段2.2：FE处理Actions，提取结构化事实（事实提取器FE的核心功能）
    
    功能：为Actionsₜ中的每个子任务pᵢ，通过"检索→分析→提取"三步，生成高保真的结构化事实。
    对应REAP论文的"子步骤2.2：FE处理Actionsₜ，提取结构化事实"。
    
    流程说明：
    1. 文档检索：调用检索器（如e5-large-v2），根据子任务查询qᵢ从语料库C中获取相关文档Dₜ
       公式：D_t = Retriever(q_t; C) （公式1）
    2. LLM分析与推理：调用LLM（如Llama-3.1-8B-Instruct），结合查询qᵢ、检索文档Dₜ、历史事实Fₜ₋₁，
       生成结构化事实fₜ
       公式：f_t = M_θ(ExtractF | q_t, D_t, F_{t-1}) （公式7）
    3. 结构化事实定义：fₜ是含4个元素的元组，格式为f_t = (s_t, e_t, r_t, l_t) （公式8）
       - sₜ：核心陈述（简洁事实断言，如"《Week Without You》的演唱者是Miley Cyrus"）
       - eₜ：直接文本证据（支撑sₜ的文档片段）
       - rₜ：推理过程（LLM推导sₜ的思维链）
       - lₜ：满足度标签（DirectAnswer/PartialClue/Failed，用于指导SP后续调度）
    
    示例（迭代1处理req1）：
        FE检索到含《Week Without You》信息的文档D₁，结合F₀=∅，生成f₁：
        - s₁：The performer of the song 'Week Without You' is Miley Cyrus.
        - e₁：Document 1："Miley Ray Hemsworth ... released the song Week Without You in 2023"
        - r₁：文档1明确提及Miley Cyrus与该歌曲的关联，无矛盾信息，可确定为演唱者
        - l₁：DirectAnswer（直接完整回答了req1）

    Args:
        search_query: 检索查询字符串，对应子任务的查询qᵢ
        requirement: 单个子任务字典，包含requirement_id、question、depends_on等字段
        collected_facts: 历史事实列表Fₜ₋₁，包含之前迭代中收集的所有事实

    Returns:
        dict: 包含"reasoned_facts"键的字典，值为结构化事实列表
        每个事实包含：
            - "statement" (str): 核心陈述sₜ
            - "direct_evidence" (list): 直接文本证据eₜ（文档片段列表）
            - "reasoning" (str): 推理过程rₜ
            - "fulfills_requirement_id" (str): 满足的子任务ID
            - "fulfillment_level" (str): 满足度标签lₜ（DIRECT_ANSWER/PARTIAL_CLUE/FAILED_EXTRACT）
        如果提取失败，返回空字典{}
    """
    print(f"\n--- Stage: Retrieving Context and Extracting Facts for search query: '{search_query}' ---")
    
    # 步骤1：文档检索 - 对应论文公式1：D_t = Retriever(q_t; C)
    # 从语料库C中检索与子任务查询q_t相关的Top-K文档
    search_hits = retrieve_context(search_query)

    # 步骤1.5：PRV 重排 - 对 top-k 做动态选择与排序，再送入事实提取（与 DynamicRAG 能力对齐）
    if search_hits and getattr(config, "USE_PRV_RERANK", False):
        search_hits = prv_reranker.rerank_documents(search_query, search_hits)
        print("--- PRV rerank applied ---")

    # 步骤2：格式化检索到的文档为字符串，供LLM分析使用
    # 将文档列表格式化为带标签的XML格式，便于LLM识别文档来源
    retrieved_documents_str = "No relevant context found."
    if search_hits:
        docs_list = []
        for i, hit in enumerate(search_hits):
            content = hit.get('contents', str(hit))
            title = hit.get('title', '')
            # 为每个文档添加ID和标题标签，便于LLM引用证据来源
            if title:
                docs_list.append(f'<document id={i+1} title="{title}">\n{content}\n</document>')
            else:
                docs_list.append(f'<document id={i+1}>\n{content}\n</document>')
        retrieved_documents_str = "\n\n".join(docs_list)
        print("--- Retrieved Context ---")
        if config.SHOW_RETRIEVED_CONTEXT:
            print(retrieved_documents_str)
        else:
            print("[Context display disabled]")
        print("-------------------------")
    else:
        print("No relevant context found.")

    # 步骤3：格式化输入数据为JSON字符串，供LLM提示词使用
    # 将当前子任务和历史事实转换为JSON格式，便于LLM理解上下文
    active_requirement_str = json.dumps(requirement, indent=2)  # 当前处理的子任务p_i
    facts_list_str = json.dumps(collected_facts, indent=2)  # 历史事实列表F_{t-1}

    # 步骤4：调用LLM进行事实提取 - 对应论文公式7：f_t = M_θ(ExtractF | q_t, D_t, F_{t-1})
    # LLM需要完成"证据定位→推理验证→线索挖掘"，避免幻觉
    try:
        facts_json_str = llm_adapter.extract_facts(
            query=search_query,  # 子任务查询q_t
            active_requirement=active_requirement_str,  # 当前子任务p_i
            known_facts=facts_list_str,  # 历史事实F_{t-1}
            retrieved_documents=retrieved_documents_str  # 检索文档D_t
        )

        # 步骤5：解析LLM返回的JSON响应，提取结构化事实
        # LLM返回的事实应包含reasoned_facts列表，每个事实对应公式8的f_t = (s_t, e_t, r_t, l_t)
        match = re.search(r'\{.*\}', facts_json_str, re.DOTALL)
        if not match:
            print("Error: Could not find a JSON object in the fact extraction response.")
            print(f"Raw Response:\n{facts_json_str}")
            return {}
        
        json_str = match.group(0)
        extracted_data = json.loads(json_str)

        # 处理LLM可能返回单个事实对象而非列表的情况
        # 如果返回的是单个事实对象，将其包装为列表格式
        if isinstance(extracted_data, dict) and "reasoned_facts" not in extracted_data:
            if "statement" in extracted_data:
                print("Info: LLM returned a single fact object. Wrapping it in the correct structure.")
                extracted_data = {"reasoned_facts": [extracted_data]}

        # 验证解析结果包含必需的"reasoned_facts"字段
        if "reasoned_facts" not in extracted_data:
            print("Error: Parsed JSON from fact extraction is missing 'reasoned_facts' key.")
            print(f"Parsed Data: {extracted_data}")
            return {}

        print("--- Fact Extraction Successful ---")
        num_facts = len(extracted_data.get('reasoned_facts', []))
        print(f"Extracted {num_facts} facts.")
        print("--------------------------------")

        # 返回提取的结构化事实列表{f₁, f₂, ..., fₖ}，k为Actions_t中子任务数量
        # 这些事实将被合并到F_t中，用于后续的计划更新
        return extracted_data

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from fact extraction response. {e}")
        print(f"Raw Response that caused error:\n{facts_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during fact extraction: {e}")
        return {}


def extract_facts_given_hits(search_query: str, requirement: dict, collected_facts: dict, search_hits: list) -> dict:
    """
    评测用：在已有 search_hits（如从 ctxs 经 E5 排序得到）上做重排与事实提取，不调用检索。
    与 retrieve_and_extract_facts 的步骤 1.5–5 一致：可选重排 → 格式化 → LLM 提取 → 解析。
    """
    if search_hits and getattr(config, "USE_PRV_RERANK", False):
        search_hits = prv_reranker.rerank_documents(search_query, search_hits)

    retrieved_documents_str = "No relevant context found."
    if search_hits:
        docs_list = []
        for i, hit in enumerate(search_hits):
            content = hit.get('contents', str(hit))
            title = hit.get('title', '')
            if title:
                docs_list.append(f'<document id={i+1} title="{title}">\n{content}\n</document>')
            else:
                docs_list.append(f'<document id={i+1}>\n{content}\n</document>')
        retrieved_documents_str = "\n\n".join(docs_list)

    active_requirement_str = json.dumps(requirement, indent=2)
    facts_list_str = json.dumps(collected_facts, indent=2)

    try:
        facts_json_str = llm_adapter.extract_facts(
            query=search_query,
            active_requirement=active_requirement_str,
            known_facts=facts_list_str,
            retrieved_documents=retrieved_documents_str,
        )
        match = re.search(r'\{.*\}', facts_json_str, re.DOTALL)
        if not match:
            return {}
        extracted_data = json.loads(match.group(0))
        if isinstance(extracted_data, dict) and "reasoned_facts" not in extracted_data and "statement" in extracted_data:
            extracted_data = {"reasoned_facts": [extracted_data]}
        if "reasoned_facts" not in extracted_data:
            return {}
        return extracted_data
    except (json.JSONDecodeError, Exception):
        return {}


def update_plan(query: str, collected_facts: dict, pending_requirements: list) -> dict:
    """
    阶段2.3：SP更新计划（Plan Updater子模块）
    
    功能：当FE提取的事实满足度标签lₜ=DirectAnswer（理想场景）时，调用Plan Updater执行计划更新。
    对应REAP论文"子步骤2.3：SP更新计划与事实"中的Plan Updater分支。
    
    核心操作：
    1. 事实替换：用新事实中的具体实体替换未完成子任务中的抽象占位符
       示例：迭代1后，req2的查询从"What is the birthday of the performer identified in req1?"
       替换为"What is the birthday of Miley Cyrus?"
    2. 计划分叉：若子任务提取到多个答案（如某歌曲有2位演唱者），则复制后续依赖子任务形成并行分支
       示例：为2位演唱者分别生成"查询生日"的子任务
    
    计划更新公式：P_t, Actions_{t+1} = SP(P_{t-1}, F_t, Q) （公式5）
    即基于新事实F_t生成下一轮计划P_t与可执行动作Actions_{t+1}
    
    示例（迭代1更新计划）：
        f₁的lₜ=DirectAnswer，SP调用Plan Updater：
        - 事实替换：将req2的查询更新为"What is the birthday of Miley Cyrus?"
        - 计划P₁={req2}（req1已完成，移除）
        - 下一轮Actions₂={req2}（req2依赖已满足）

    Args:
        query: 原始用户查询Q
        collected_facts: 当前收集的事实列表F_t，包含所有已提取的结构化事实
        pending_requirements: 待完成的子任务列表，对应任务计划P_{t-1}中未完成的部分

    Returns:
        dict: 包含"decision"键的字典，decision包含：
            - "next_step" (str): 下一步动作，为"SYNTHESIZE_ANSWER"或"CONTINUE_SEARCH"
            - "updated_plan" (list): 更新后的任务计划P_t（如果next_step为CONTINUE_SEARCH）
            - "next_actions" (list): 下一轮可执行动作列表Actions_{t+1}（如果next_step为CONTINUE_SEARCH）
        如果更新失败，返回空字典{}
    """
    print(f"\n--- Stage: Updating Plan for query: '{query}' ---")
    
    # 步骤1：格式化输入数据为JSON字符串，供LLM提示词使用
    # 将当前事实列表和待完成子任务转换为JSON格式
    collected_facts_str = json.dumps(collected_facts, indent=2)  # 当前事实列表F_t
    pending_requirements_str = json.dumps(pending_requirements, indent=2)  # 待完成计划P_{t-1}

    # 步骤2：调用LLM适配器进行计划更新
    # LLM将根据SYSTEM_PROMPT_PLAN_UPDATER提示词，执行事实替换和计划分叉
    try:
        update_json_str = llm_adapter.update_plan(
            query=query,  # 原始查询Q
            collected_facts=collected_facts_str,  # 当前事实F_t
            pending_requirements=pending_requirements_str  # 待完成计划P_{t-1}
        )

        # 步骤3：解析LLM返回的JSON响应，提取计划更新结果
        # LLM返回的决策应包含next_step、updated_plan和next_actions字段
        match = re.search(r'\{.*\}', update_json_str, re.DOTALL)
        if not match:
            print("Error: Could not find a JSON object in the plan update response.")
            print(f"Raw Response:\n{update_json_str}")
            return {}
        
        json_str = match.group(0)
        update_result = json.loads(json_str)

        # 步骤4：验证解析结果包含必需的"decision"字段
        if "decision" not in update_result:
            print("Error: Parsed JSON from plan update is missing 'decision' key.")
            print(f"Parsed Data: {update_result}")
            return {}

        print("--- Plan Update Successful ---")
        decision = update_result.get("decision", {})
        next_step = decision.get("next_step", "UNKNOWN")
        print(f"Decision: {next_step}")
        if next_step == "CONTINUE_SEARCH":
            next_actions = decision.get("next_actions", [])
            print(f"Next Actions: {len(next_actions)} new search(es) planned.")
        print("----------------------------")

        # 返回计划更新结果，包含下一轮的计划P_t和可执行动作Actions_{t+1}
        return update_result

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from plan update response. {e}")
        print(f"Raw Response that caused error:\n{update_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during plan update: {e}")
        return {}

def replan_questions(query: str, collected_facts: dict, pending_requirements: list) -> dict:
    """
    阶段2.3：SP更新计划（Re-Planner子模块）
    
    功能：当FE提取的事实满足度标签lₜ=PartialClue/Failed（非理想场景）时，调用Re-Planner执行计划修复。
    对应REAP论文"子步骤2.3：SP更新计划与事实"中的Re-Planner分支。
    
    核心操作：
    1. 实用充分性评估：判断部分线索是否足够推进后续推理
       示例：查询"某导演代表作"时，仅提取到1部作品，但该作品已能回答原始问题，则视为"足够"，
       无需继续检索
    2. 范围化计划修复：若线索不足，先诊断问题类型，然后执行修复
       - 局部问题：子查询表述模糊 → 优化查询（如将"When did he die?"优化为"When did Giuliano Carnimeo die?"）
       - 系统性问题：推理路径错误 → 修剪无效分支+注入新子任务
    
    计划更新公式：P_t, Actions_{t+1} = SP(P_{t-1}, F_t, Q) （公式5）
    即基于新事实F_t生成下一轮计划P_t与可执行动作Actions_{t+1}
    
    示例（文档1-50案例）：
        若"查询导演生日"提取失败，Re-Planner可能将子查询从"When did he die?"
        优化为"When did Giuliano Carnimeo die?"（补充姓名）

    Args:
        query: 原始用户查询Q
        collected_facts: 当前收集的事实列表F_t，包含所有已提取的结构化事实（可能包含PartialClue）
        pending_requirements: 待完成的子任务列表，对应任务计划P_{t-1}中未完成的部分

    Returns:
        dict: 包含以下键的字典：
            - "analysis" (dict): 问题诊断和恢复策略分析
                - "problem_diagnosis" (str): 问题范围的诊断（局部问题/系统性问题）
                - "recovery_strategy" (str): 选择的恢复策略摘要
            - "decision" (dict): 决策信息
                - "next_step" (str): 下一步动作，为"SYNTHESIZE_ANSWER"或"CONTINUE_SEARCH"
                - "updated_plan" (list): 修复后的任务计划P_t
                - "next_actions" (list): 下一轮可执行动作列表Actions_{t+1}
        如果重新规划失败，返回空字典{}
    """
    print(f"\n--- Stage: Replanning Next Actions for query: '{query}' ---")
    
    # 步骤1：格式化输入数据为JSON字符串，供LLM提示词使用
    # 将当前事实列表和待完成子任务转换为JSON格式
    collected_facts_str = json.dumps(collected_facts, indent=2)  # 当前事实列表F_t（可能包含PartialClue）
    pending_requirements_str = json.dumps(pending_requirements, indent=2)  # 待完成计划P_{t-1}

    # 步骤2：调用LLM适配器进行计划重新规划
    # LLM将根据SYSTEM_PROMPT_CONDITION_REPLAN提示词，执行实用充分性评估和范围化计划修复
    try:
        replan_json_str = llm_adapter.replan_conditions(
            query=query,  # 原始查询Q
            collected_facts=collected_facts_str,  # 当前事实F_t
            pending_requirements=pending_requirements_str  # 待完成计划P_{t-1}
        )

        # 步骤3：解析LLM返回的JSON响应，提取重新规划结果
        # LLM返回的结果应包含analysis（问题诊断）和decision（决策）两个字段
        match = re.search(r'\{.*\}', replan_json_str, re.DOTALL)
        if not match:
            print("Error: Could not find a JSON object in the replanning response.")
            print(f"Raw Response:\n{replan_json_str}")
            return {}
        
        json_str = match.group(0)
        replan_result = json.loads(json_str)

        # 步骤4：验证解析结果包含必需的"analysis"和"decision"字段
        if "analysis" not in replan_result or "decision" not in replan_result:
            print("Error: Parsed JSON from replanning is missing 'analysis' or 'decision' key.")
            print(f"Parsed Data: {replan_result}")
            return {}

        print("--- Replanning Successful ---")
        decision = replan_result.get("decision", {})
        next_step = decision.get("next_step", "UNKNOWN")
        print(f"Decision: {next_step}")
        if next_step == "CONTINUE_SEARCH":
            next_actions = decision.get("next_actions", [])
            print(f"Next Actions: {len(next_actions)} new search(es) planned.")
        print("---------------------------")

        # 返回重新规划结果，包含问题诊断、恢复策略和修复后的计划P_t
        return replan_result

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from replanning response. {e}")
        print(f"Raw Response that caused error:\n{replan_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during replanning: {e}")
        return {}

def synthesize_final_answer(query: str, collected_facts: dict) -> str:
    """
    阶段3：答案合成（Synthesizer模块）
    
    功能：调用LLM，基于最终事实列表F_final中的所有事实（含核心陈述、证据、推理过程），
    合成符合原始查询需求的最终答案，确保答案的完整性与可追溯性。
    对应REAP论文"阶段3：答案合成"。
    
    合成公式：A = M_θ(Synthesize | Q, F_final) （公式4）
    其中F_final包含所有迭代中收集的结构化事实，每个事实包含：
    - sₜ：核心陈述
    - eₜ：直接文本证据
    - rₜ：推理过程
    
    示例（文档1-276案例）：
        Synthesizer结合"导演是Giuliano Carnimeo"与"其死亡日期为2016年9月10日"，
        合成最终答案"10 September 2016"
    
    关键要求：
    - 答案必须基于F_final中的事实，不能使用外部知识
    - 答案应简洁、直接，避免冗余信息
    - 答案应完整回答原始查询Q

    Args:
        query: 原始用户查询Q
        collected_facts: 最终事实列表F_final，包含所有迭代中收集的结构化事实
            格式为{"reasoned_facts": [f₁, f₂, ..., fₙ]}，每个fᵢ对应公式8的f_t = (s_t, e_t, r_t, l_t)

    Returns:
        str: 原始查询的最终答案A，如果生成失败则返回错误消息字符串
    """
    print(f"\n--- Stage: Synthesizing Final Answer for query: '{query}' ---")

    # 步骤1：格式化收集的事实为JSON字符串，供LLM提示词使用
    # 将最终事实列表F_final转换为JSON格式，包含所有结构化事实的完整信息
    facts_str = json.dumps(collected_facts, indent=2)

    # 步骤2：调用LLM适配器生成最终答案 - 对应论文公式4：A = M_θ(Synthesize | Q, F_final)
    # LLM将根据SYSTEM_PROMPT_FINAL_ANSWER提示词，基于所有事实合成最终答案
    try:
        final_answer = llm_adapter.generate_final_answer(
            query=query,  # 原始查询Q
            facts=facts_str  # 最终事实列表F_final
        )

        print("--- Final Answer Generation Successful ---")
        # 最终答案由调用函数打印，这里直接返回
        return final_answer

    except Exception as e:
        error_message = f"An unexpected error occurred during final answer synthesis: {e}"
        print(error_message)
        return error_message

