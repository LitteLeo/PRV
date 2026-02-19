import config
from openai import OpenAI
import httpx
from .. import prompts

def configure_openai_client():
    """
    Configures and returns the OpenAI API client.
    Checks for OPENAI_API_KEY in config.py and applies proxy if configured.
    """
    if not hasattr(config, 'OPENAI_API_KEY') or not config.OPENAI_API_KEY:
        raise ValueError("Error: OPENAI_API_KEY is not set in config.py.")

    client_params = {
        "api_key": config.OPENAI_API_KEY,
        "base_url": "baseurl" 
    }

    if config.PROXY_ENABLED and config.PROXY_URL:
        proxy = config.PROXY_URL
        _http_client = httpx.Client(proxy=proxy)
        client_params["http_client"] = _http_client
        print(f"OpenAI API client will use proxy: {config.PROXY_URL}")
    else:
        print("OpenAI API client will not use proxy.")

    try:
        client = OpenAI(**client_params)
        print("OpenAI API client configured.")
        return client
    except Exception as e:
        print(f"Error configuring OpenAI API client: {e}")
        raise

def openai_generate_rag_response(query: str, context: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Generates a response using the OpenAI LLM based on the query and context."""
    user_prompt = prompts.USER_PROMPT_TEMPLATE_RAG.format(context=context, query=query)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to generate RAG response...")
        
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_RAG},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty RAG response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to generate RAG response: {e}")
        return f"Sorry, encountered an error when calling LLM to generate RAG response: {e}"

def openai_analyze_query(query: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Analyzes the input query using openai."""
    combined_instruction = prompts.USER_PROMPT_QUERY_ANALYSIS.format(query=query)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to analyze query...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_QUERY_ANALYSIS},
                {"role": "user", "content": combined_instruction}
            ]
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty query analysis response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI ({llm_model_name}) to analyze query: {e}")
        return f"Sorry, encountered an error when calling LLM to analyze query: {e}"


def openai_extract_facts(query: str, active_requirement: str, retrieved_documents: str,  known_facts: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Extracts facts from the context that satisfy the given condition based on the query using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_FACT_EXTRACTION.format(query=query, active_requirement=active_requirement, known_facts=known_facts, retrieved_documents=retrieved_documents)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to extract facts from context...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_FACT_EXTRACTION},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty fact extraction response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to extract facts from context: {e}")
        return f"Sorry, encountered an error when calling LLM to extract facts: {e}"

def openai_update_plan(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Updates the plan based on collected facts and pending requirements using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_PLAN_UPDATER.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to update plan...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_PLAN_UPDATER},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty plan update response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to update plan: {e}")
        return f"Sorry, encountered an error when calling LLM to update plan: {e}"

def openai_replan_conditions(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Replans the required conditions based on the user query, initial condition, and extracted facts using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_CONDITION_REPLAN.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to replan conditions...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_CONDITION_REPLAN},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty condition replan response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to replan conditions: {e}")
        return f"Sorry, encountered an error when calling LLM to replan conditions: {e}"

def openai_generate_final_answer(query: str, facts: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Generates the final answer based on the user query and extracted facts using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_FINAL_ANSWER.format(query=query, facts=facts)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to generate final answer...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_FINAL_ANSWER},
                {"role": "user", "content": user_prompt}
            ]
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty final answer response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to generate final answer: {e}")
        return f"Sorry, encountered an error when calling LLM to generate final answer: {e}"

def openai_evaluate_answer(question: str, golden_answer: str, predicted_answer: str, llm_model_name: str = config.LLM_MODEL_EVAL) -> str:
    """
    Uses OpenAI LLM to evaluate if the predicted answer is correct.
    Returns "True", "False", or an error string.
    """
    user_prompt = prompts.USER_PROMPT_EVALUATION.format(
        question=question,
        golden_answer=golden_answer,
        predicted_answer=predicted_answer
    )

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to evaluate answer...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_EVALUATION},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5,
            temperature=0.0
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Evaluation Error: Empty response from OpenAI."
        
        cleaned_response = response_text.strip().lower()
        if "true" in cleaned_response:
            return "True"
        elif "false" in cleaned_response:
            return "False"
        else:
            return f"Evaluation Error: Unexpected response '{response_text.strip()}'"
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to evaluate answer: {e}")
        return f"Evaluation Error: {e}"