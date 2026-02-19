"""
Configuration settings for the RAG pipeline.
Loads sensitive keys from .env file.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- AI Provider Selection (for LLM Generation) ---
# Supported: 'ollama', 'gemini', 'grok', 'deepseek', 'vllm', 'openai'
# AI_PROVIDER = os.getenv('AI_PROVIDER', 'vllm').lower()
AI_PROVIDER = 'vllm'

# --- API Keys (Loaded from .env file or environment variables) ---

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # For OpenAI


# --- VLLM Settings ---
# VLLM_HOST = os.getenv('VLLM_HOST', "http://172.18.144.20")
VLLM_HOST = os.getenv('VLLM_HOST', "http://127.0.0.1")


# --- PRV 统一模型：REAP-all-lora（DynamicRAG-8B + LoRA，规划/重排/校验同一模型）---
# Default model and port for general RAG response generation
VLLM_LLM_MODEL = os.getenv("VLLM_LLM_MODEL", "/home/lfy/projects/models/REAP-all-lora")
VLLM_LLM_PORT = int(os.getenv("VLLM_LLM_PORT", 8000))

# --- VLLM Dedicated Models Configuration ---
# PRV 单模型：所有任务使用同一 REAP-all-lora，设为 False。
VLLM_USE_DEDICATED_MODELS = False

# 各任务模型配置（单模型时均使用默认 VLLM_LLM_MODEL / VLLM_LLM_PORT）
VLLM_ANALYZE_QUERY_MODEL = VLLM_LLM_MODEL
VLLM_ANALYZE_QUERY_PORT = VLLM_LLM_PORT

VLLM_EXTRACT_FACTS_MODEL = VLLM_LLM_MODEL
VLLM_EXTRACT_FACTS_PORT = VLLM_LLM_PORT

VLLM_UPDATE_PLAN_MODEL = VLLM_LLM_MODEL
VLLM_UPDATE_PLAN_PORT = VLLM_LLM_PORT

VLLM_REPLAN_CONDITIONS_MODEL = VLLM_LLM_MODEL
VLLM_REPLAN_CONDITIONS_PORT = VLLM_LLM_PORT

VLLM_GENERATE_FINAL_ANSWER_MODEL = VLLM_LLM_MODEL
VLLM_GENERATE_FINAL_ANSWER_PORT = VLLM_LLM_PORT


# Deprecated, use VLLM_HOST and specific ports instead
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', f"{VLLM_HOST}:{VLLM_LLM_PORT}/v1")

OPENAI_LLM_MODEL = "gpt-4"


# --- LLM for evaluation ---
LLM_MODEL_EVAL = "gpt-4"

# --- Search Service Configuration ---
# SEARCH_SERVICE_HOST = os.getenv('SEARCH_SERVICE_HOST', '172.18.144.21')
# SEARCH_SERVICE_PORT = int(os.getenv('SEARCH_SERVICE_PORT', 8090))
SEARCH_SERVICE_HOST = '127.0.0.1'
SEARCH_SERVICE_PORT = 8090

# --- Proxy Configuration (Optional) ---
# Set PROXY_ENABLED to 'True' or 'true' in .env or environment to enable
PROXY_ENABLED_str = os.getenv('PROXY_ENABLED', 'False').lower()
PROXY_ENABLED = PROXY_ENABLED_str == 'true'
PROXY_URL = os.getenv('PROXY_URL', None) 


# --- RAG Parameters ---
TOP_K = int(os.getenv('TOP_K', 5)) # Number of documents to retrieve

# --- Control Flags ---
SHOW_RETRIEVED_CONTEXT = True  # Set to False to hide context printout

# PRV 重排：召回后是否对 top-k 做模型动态选择与排序（再送入事实提取）
USE_PRV_RERANK = os.getenv("USE_PRV_RERANK", "true").lower() == "true"

# 评测时「仅从 ctxs 检索」使用的 E5 编码器路径（进程内排序，不依赖 E5 检索服务）
E5_ENCODER_PATH = os.getenv("E5_ENCODER_PATH", "/home/lfy/projects/models/e5-large-v2")

# --- LLM Model Name (Set based on provider for generation) ---
LLM_MODEL_NAME = None # Initialize
if  AI_PROVIDER == 'vllm':
    LLM_MODEL_NAME = VLLM_LLM_MODEL
elif AI_PROVIDER == 'openai':
    LLM_MODEL_NAME = OPENAI_LLM_MODEL
else:
    raise ValueError(f"Unsupported AI_PROVIDER for LLM: {AI_PROVIDER}.")


