"""
REAP框架LLM适配器模块（llm_adapter.py）

本模块提供统一的LLM调用接口，支持多种LLM提供商（vLLM、OpenAI等），
并自动记录所有LLM调用用于追踪和调试。

功能：
1. 统一的LLM调用接口：封装不同提供商的差异
2. 自动追踪：记录所有LLM调用的输入和输出
3. 提供商配置：根据config.AI_PROVIDER选择对应的提供商

REAP框架中的LLM调用：
1. analyze_query：查询分解（Decomposer模块）
2. extract_facts：事实提取（FE模块）
3. update_plan：计划更新（Plan Updater子模块）
4. replan_conditions：计划重新规划（Re-Planner子模块）
5. generate_final_answer：答案合成（Synthesizer模块）
"""
import sys
import time
import config
from rag_pipeline_lib.llm_providers import vllm_utils, openai_utils
from rag_pipeline_lib.pipeline import tracer_context
from functools import wraps

def traceable_llm_call(func):
    """
    装饰器：自动追踪LLM调用，并记录单次调用耗时 duration_s（秒），用于 Latency 实验。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = tracer_context.get()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        duration_s = time.perf_counter() - t0
        if tracer:
            input_args = {key: value for key, value in kwargs.items()}
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            for i, arg in enumerate(args):
                if i < len(arg_names):
                    input_args[arg_names[i]] = arg
            tracer.record_llm_call(
                adapter_function_name=func.__name__,
                inputs=input_args,
                output=result,
                duration_s=duration_s,
            )
        return result
    return wrapper

def configure_llm_provider():
    print(f"Configuring LLM provider: {config.AI_PROVIDER}")
    if config.AI_PROVIDER == 'vllm':
        try:
            vllm_utils.configure_vllm_client()
        except Exception as e:
            print(f"Error: Failed to configure vLLM: {e}", file=sys.stderr)
            raise
    elif config.AI_PROVIDER == 'openai':
        try:
            openai_utils.configure_openai_client()
        except Exception as e:
            print(f"Error: Failed to configure OpenAI: {e}", file=sys.stderr)
            raise
    else:
        error_msg = f"Unsupported AI_PROVIDER: {config.AI_PROVIDER}. Please choose 'gemini', 'ollama', 'grok', 'deepseek', 'vllm', or 'openai'."
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg)

@traceable_llm_call
def generate_rag_response(query: str, context: str) -> str:
    """
    Generates a RAG response using the configured LLM provider.
    The default model from config.py for the provider will be used.
    """
    if config.AI_PROVIDER == 'vllm':
        return vllm_utils.vllm_generate_rag_response(query, context)
    elif config.AI_PROVIDER == 'openai':
        return openai_utils.openai_generate_rag_response(query, context)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {config.AI_PROVIDER}. Cannot generate RAG response.")

@traceable_llm_call
def analyze_query(query: str) -> str:
    """
    查询分解：调用LLM将复杂查询Q拆解为结构化初始任务计划P₀
    
    对应REAP阶段1：Decomposer模块
    对应函数：rag_core.analyze_and_decompose_query
    使用提示词：SYSTEM_PROMPT_QUERY_ANALYSIS
    
    Args:
        query: 用户的复杂多跳查询Q
        
    Returns:
        str: LLM返回的JSON字符串，包含user_goal和requirements列表
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling analyze_query (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_analyze_query(query)
    elif provider == 'openai':
        return openai_utils.openai_analyze_query(query)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot analyze query.")

@traceable_llm_call
def rerank_documents(query: str, retrieved_content: str) -> str:
    """
    PRV 重排：调用同一模型对检索文档做动态选择与排序（文档标识输出）。
    对应 PRV 框架「重排」支柱，与 DynamicRAG 文档选择能力对齐。
    """
    provider = config.AI_PROVIDER
    if provider == "vllm":
        return vllm_utils.vllm_rerank_documents(query, retrieved_content)
    # 非 vLLM 时返回空，调用方将保留原文档列表
    return ""

@traceable_llm_call
def extract_facts(query: str, active_requirement: str, known_facts: str, retrieved_documents: str) -> str:
    """
    事实提取：调用LLM从检索文档中提取结构化事实f_t = (s_t, e_t, r_t, l_t)
    
    对应REAP阶段2.2：FE模块
    对应函数：rag_core.retrieve_and_extract_facts
    对应公式：f_t = M_θ(ExtractF | q_t, D_t, F_{t-1}) （公式7）
    使用提示词：SYSTEM_PROMPT_FACT_EXTRACTION
    
    Args:
        query: 原始用户查询Q
        active_requirement: 当前处理的子任务p_i（JSON字符串）
        known_facts: 历史事实列表F_{t-1}（JSON字符串）
        retrieved_documents: 检索到的文档D_t（格式化字符串）
        
    Returns:
        str: LLM返回的JSON字符串，包含reasoned_facts列表
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling extract_facts (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_extract_facts(query, active_requirement, known_facts, retrieved_documents)
    elif provider == 'openai':
        return openai_utils.openai_extract_facts(query, active_requirement, known_facts, retrieved_documents)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot extract facts.")

@traceable_llm_call
def update_plan(query: str, collected_facts: str, pending_requirements: str) -> str:
    """
    计划更新：调用LLM执行事实替换和计划分叉（Plan Updater子模块）
    
    对应REAP阶段2.3：Plan Updater子模块（当lₜ=DirectAnswer时使用）
    对应函数：rag_core.update_plan
    对应公式：P_t, Actions_{t+1} = SP(P_{t-1}, F_t, Q) （公式5）
    使用提示词：SYSTEM_PROMPT_PLAN_UPDATER
    
    Args:
        query: 原始用户查询Q
        collected_facts: 当前事实列表F_t（JSON字符串）
        pending_requirements: 待完成计划P_{t-1}（JSON字符串）
        
    Returns:
        str: LLM返回的JSON字符串，包含decision对象（next_step、updated_plan、next_actions）
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling update_plan (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_update_plan(query, collected_facts, pending_requirements)
    elif provider == 'openai':
        return openai_utils.openai_update_plan(query, collected_facts, pending_requirements)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot update plan.")

@traceable_llm_call
def replan_conditions(query: str, collected_facts: str, pending_requirements: str) -> str:
    """
    计划重新规划：调用LLM执行实用充分性评估和范围化计划修复（Re-Planner子模块）
    
    对应REAP阶段2.3：Re-Planner子模块（当lₜ=PartialClue/Failed时使用）
    对应函数：rag_core.replan_questions
    对应公式：P_t, Actions_{t+1} = SP(P_{t-1}, F_t, Q) （公式5）
    使用提示词：SYSTEM_PROMPT_CONDITION_REPLAN
    
    Args:
        query: 原始用户查询Q
        collected_facts: 当前事实列表F_t（可能包含PartialClue，JSON字符串）
        pending_requirements: 待完成计划P_{t-1}（JSON字符串）
        
    Returns:
        str: LLM返回的JSON字符串，包含analysis（问题诊断）和decision（决策）对象
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling replan_conditions (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_replan_conditions(query, collected_facts, pending_requirements)
    elif provider == 'openai':
        return openai_utils.openai_replan_conditions(query, collected_facts, pending_requirements)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot replan conditions.")

@traceable_llm_call
def generate_final_answer(query: str, facts: str) -> str:
    """
    答案合成：调用LLM基于最终事实列表F_final合成最终答案A
    
    对应REAP阶段3：Synthesizer模块
    对应函数：rag_core.synthesize_final_answer
    对应公式：A = M_θ(Synthesize | Q, F_final) （公式4）
    使用提示词：SYSTEM_PROMPT_FINAL_ANSWER
    
    Args:
        query: 原始用户查询Q
        facts: 最终事实列表F_final（JSON字符串）
        
    Returns:
        str: 原始查询的最终答案A
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling generate_final_answer (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_generate_final_answer(query, facts)
    elif provider == 'openai':
        return openai_utils.openai_generate_final_answer(query, facts)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot generate final answer.")


@traceable_llm_call
def generate_final_answer_fever(query: str, facts: str) -> str:
    """
    FEVER 专用答案合成：调用LLM将最终事实列表映射为 SUPPORTS / REFUTES / NOT ENOUGH INFO 三类标签之一。
    
    对应：synthesize_final_answer_fever
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling generate_final_answer_fever (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_generate_final_answer_fever(query, facts)
    elif provider == 'openai':
        # 如需支持 openai，可在 openai_utils 中实现对应函数
        raise ValueError("OpenAI provider is not yet supported for FEVER-specific final answer generation.")
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot generate FEVER final answer.")

@traceable_llm_call
def generate_final_answer_with_style(query: str, facts: str, style_hint: str) -> str:
    """
    带风格提示的答案合成：让 LLM 按短短一句 / 段落等风格输出。
    用于 NQ / TriviaQA / PopQA / 2wikimqa / HotpotQA / ASQA / ELI5 等任务特化。
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling generate_final_answer_with_style (AI Provider: {provider}, style='{style_hint}')")
    if provider == 'vllm':
        return vllm_utils.vllm_generate_final_answer_with_style(query, facts, style_hint)
    elif provider == 'openai':
        # 如需支持 openai，可在 openai_utils 中实现对应函数
        raise ValueError("OpenAI provider is not yet supported for style-specific final answer generation.")
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot generate styled final answer.")

@traceable_llm_call
def evaluate_answer(question: str, golden_answer: str, predicted_answer: str) -> str:
    """
    Evaluates if the predicted answer is correct using the configured LLM provider.
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling evaluate_answer (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_evaluate_answer(question, golden_answer, predicted_answer)
    elif provider == 'openai':
        return openai_utils.openai_evaluate_answer(question, golden_answer, predicted_answer)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot evaluate answer.")