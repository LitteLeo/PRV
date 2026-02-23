import config
from openai import OpenAI
import httpx  # Required for proxy support
from .. import prompts
import time

def configure_vllm_client(task_type: str = "rag_response"):
    """
    Configures and returns the OpenAI-compatible client for vLLM.
    Selects the base URL and model based on the task type and config settings.

    Args:
        task_type (str): The type of task being performed. 
                         Examples: 'rag_response', 'analyze_query', 'extract_facts', etc.
    """
    model_name = config.VLLM_LLM_MODEL
    port = config.VLLM_LLM_PORT

    if config.VLLM_USE_DEDICATED_MODELS and task_type not in ["rag_response", "evaluate_answer", "rerank_documents"]:
        model_map = {
            "analyze_query": (config.VLLM_ANALYZE_QUERY_MODEL, config.VLLM_ANALYZE_QUERY_PORT),
            "extract_facts": (config.VLLM_EXTRACT_FACTS_MODEL, config.VLLM_EXTRACT_FACTS_PORT),
            "update_plan": (config.VLLM_UPDATE_PLAN_MODEL, config.VLLM_UPDATE_PLAN_PORT),
            "replan_conditions": (config.VLLM_REPLAN_CONDITIONS_MODEL, config.VLLM_REPLAN_CONDITIONS_PORT),
            "generate_final_answer": (config.VLLM_GENERATE_FINAL_ANSWER_MODEL, config.VLLM_GENERATE_FINAL_ANSWER_PORT),
        }
        if task_type in model_map:
            model_name, port = model_map[task_type]
            print(f"Using dedicated VLLM model for '{task_type}': {model_name} on port {port}")
        else:
            print(f"Warning: No dedicated model found for task '{task_type}'. Falling back to default.")

    base_url_to_use = f"{config.VLLM_HOST}:{port}/v1"
    
    print(f"Configuring vLLM client for '{task_type}': URL='{base_url_to_use}'")

    # 创建带超时的 HTTP 客户端
    http_client = httpx.Client(
        timeout=httpx.Timeout(180.0, connect=10.0),  # 总超时180秒，连接超时10秒
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )

    client_params = {
        "api_key": "empty",
        "base_url": base_url_to_use,
        "timeout":180.0, # OpenAI 客户段超时(秒）
    }


    try:
        client = OpenAI(**client_params)
        print("VLLM API client configured successfully.")
        return client, model_name
    except Exception as e:
        print(f"Error configuring VLLM API client: {e}")
        raise

def vllm_generate_rag_response(query: str, context: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Generates a response using VLLM based on the query and context."""
    # Combine system prompt and user prompt into a single instruction
    combined_instruction = f"{prompts.SYSTEM_PROMPT_RAG}\n{prompts.USER_PROMPT_TEMPLATE_RAG.format(context=context, query=query)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="rag_response")
        print(f"Calling VLLM ({model_to_use}) to generate RAG response...")
        
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0.2
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty RAG response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to generate RAG response: {e}")
        return f"Sorry, encountered an error when calling LLM to generate RAG response: {e}"

def vllm_analyze_query(query: str, llm_model_name: str = None) -> str:
    """Analyzes the input query using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_QUERY_ANALYSIS}\n{prompts.USER_PROMPT_QUERY_ANALYSIS.format(query=query)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="analyze_query")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to analyze query...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0.2
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty query analysis response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to analyze query: {e}")
        return f"Sorry, encountered an error when calling LLM to analyze query: {e}"

def vllm_extract_facts(query: str, active_requirement: str, retrieved_documents: str, known_facts: str, llm_model_name: str = None) -> str:
    """Extracts facts from the context that satisfy the given condition based on the query using VLLM."""
    if getattr(config, "USE_VERIFICATION_CONSTRAINTS", True):
        sys_prompt = prompts.SYSTEM_PROMPT_FACT_EXTRACTION
    else:
        sys_prompt = prompts.SYSTEM_PROMPT_FACT_EXTRACTION_NO_VERIFICATION
    combined_instruction = f"{sys_prompt}\n{prompts.USER_PROMPT_FACT_EXTRACTION.format(query=query, active_requirement=active_requirement, known_facts=known_facts, retrieved_documents=retrieved_documents)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="extract_facts")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to extract facts from context...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0.2
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty fact extraction response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to extract facts from context: {e}")
        return f"Sorry, encountered an error when calling LLM to extract facts: {e}"

def vllm_update_plan(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = None) -> str:
    """Updates the plan based on collected facts and pending requirements using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_PLAN_UPDATER}\n{prompts.USER_PROMPT_PLAN_UPDATER.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)}"

    max_retries = 2
    retry_delay = 3

    for attempt in range(max_retries):
        try:
            client, model_to_use = configure_vllm_client(task_type="update_plan")
            if llm_model_name is None: llm_model_name = model_to_use
            if attempt > 0:
                print(f"Calling VLLM ({llm_model_name}) to update plan... (retry {attempt + 1}/{max_retries})")
            else:
                print(f"Calling VLLM ({llm_model_name}) to update plan...")
            
            completion = client.chat.completions.create(
                model=llm_model_name,
                messages=[
                    {"role": "user", "content": combined_instruction}
                ],
                temperature=0.2,
                timeout=180.0  # 添加超时
            )
            response_text = completion.choices[0].message.content
            if not response_text:
                return "Empty plan update response received from VLLM API."
            return response_text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error calling VLLM (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"Error calling VLLM ({llm_model_name}) to update plan after {max_retries} attempts: {e}")
                return f"Sorry, encountered an error when calling LLM to update plan: {e}"

    #try:
    #    client, model_to_use = configure_vllm_client(task_type="update_plan")
    #    if llm_model_name is None: llm_model_name = model_to_use
    #    print(f"Calling VLLM ({llm_model_name}) to update plan...")
    #    completion = client.chat.completions.create(
    #        model=llm_model_name,
    #        messages=[
    #            {"role": "user", "content": combined_instruction}
    #        ],
    #        temperature=0.2
    #    )
    #    response_text = completion.choices[0].message.content
    #    if not response_text:
    #        return "Empty plan update response received from VLLM API."
    #    return response_text.strip()
    #except Exception as e:
    #    print(f"Error calling VLLM ({llm_model_name}) to update plan: {e}")
    #    return f"Sorry, encountered an error when calling LLM to update plan: {e}"

def vllm_replan_conditions(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = None) -> str:
    """Replans the required conditions based on the user query, initial condition, and extracted facts using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_CONDITION_REPLAN}\n{prompts.USER_PROMPT_CONDITION_REPLAN.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="replan_conditions")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to replan conditions...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0.2
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty condition replan response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to replan conditions: {e}")
        return f"Sorry, encountered an error when calling LLM to replan conditions: {e}"

def vllm_generate_final_answer(query: str, facts: str, llm_model_name: str = None) -> str:
    """Generates the final answer based on the user query and extracted facts using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_FINAL_ANSWER}\n{prompts.USER_PROMPT_FINAL_ANSWER.format(query=query, facts=facts)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="generate_final_answer")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to generate final answer...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0.2
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty final answer response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to generate final answer: {e}")
        return f"Sorry, encountered an error when calling LLM to generate final answer: {e}"


def vllm_generate_final_answer_fever(query: str, facts: str, llm_model_name: str = None) -> str:
    """
    FEVER 专用：基于 query 和事实列表生成 SUPPORTS / REFUTES / NOT ENOUGH INFO 三类标签之一。
    使用 SYSTEM_PROMPT_FINAL_ANSWER_FEVER / USER_PROMPT_FINAL_ANSWER_FEVER 提示词。
    """
    combined_instruction = f"{prompts.SYSTEM_PROMPT_FINAL_ANSWER_FEVER}\n{prompts.USER_PROMPT_FINAL_ANSWER_FEVER.format(query=query, facts=facts)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="generate_final_answer")
        if llm_model_name is None:
            llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to generate FEVER label...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            max_tokens=8,
            temperature=0.0,
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "EMPTY"
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to generate FEVER label: {e}")
        return f"ERROR: {e}"


def vllm_generate_final_answer_with_style(
    query: str,
    facts: str,
    style_hint: str,
    llm_model_name: str = None,
) -> str:
    """
    根据 query + facts + 风格提示生成最终答案，用于 NQ / TriviaQA / PopQA / 2wikimqa /
    HotpotQA / ASQA / ELI5 等任务特化（短短一句 / 段落）。
    """
    combined_instruction = (
        f"{prompts.SYSTEM_PROMPT_FINAL_ANSWER}\n"
        f"Instruction: {style_hint}\n"
        f"{prompts.USER_PROMPT_FINAL_ANSWER.format(query=query, facts=facts)}"
    )

    try:
        client, model_to_use = configure_vllm_client(task_type="generate_final_answer")
        if llm_model_name is None:
            llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to generate final answer with style hint...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[{"role": "user", "content": combined_instruction}],
            temperature=0.2,
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty final answer response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to generate final answer with style: {e}")
        return f"Sorry, encountered an error when calling LLM to generate final answer with style: {e}"

def vllm_evaluate_answer(question: str, golden_answer: str, predicted_answer: str, llm_model_name: str = None) -> str:
    """
    Uses VLLM to evaluate if the predicted answer is correct.
    Returns "True", "False", or an error string.
    """
    # VLLM often works best with a single combined prompt
    instruction = f"{prompts.SYSTEM_PROMPT_EVALUATION}\n\n{prompts.USER_PROMPT_EVALUATION.format(question=question, golden_answer=golden_answer, predicted_answer=predicted_answer)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="evaluate_answer")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to evaluate answer...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": instruction}
            ],
            max_tokens=5,
            temperature=0.2
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Evaluation Error: Empty response from VLLM."
        
        cleaned_response = response_text.strip().lower()
        if "true" in cleaned_response:
            return "True"
        elif "false" in cleaned_response:
            return "False"
        else:
            return f"Evaluation Error: Unexpected response '{response_text.strip()}'"
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to evaluate answer: {e}")
        return f"Evaluation Error: {e}"


def vllm_rerank_documents(query: str, retrieved_content: str, llm_model_name: str = None) -> str:
    """
    PRV 重排：调用同一模型对检索文档做动态选择与排序。
    输入 query + 文档内容字符串，输出模型生成的文档标识序列（如 [1], [2], [5] 或 None）。
    与 DynamicRAG 的 document selection 能力对齐。
    """
    combined_instruction = (
        f"{prompts.SYSTEM_PROMPT_RERANK_DOCS}\n"
        f"{prompts.USER_PROMPT_RERANK_DOCS.format(query=query, retrieved_content=retrieved_content)}"
    )
    try:
        client, model_to_use = configure_vllm_client(task_type="rerank_documents")
        if llm_model_name is None:
            llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) for PRV rerank (document selection)...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[{"role": "user", "content": combined_instruction}],
            temperature=0.2,
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return ""
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) for rerank: {e}")
        return ""

